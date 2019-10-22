
FROM python:3.7

# Update package lists and install some software.
RUN apt-get update && apt-get install -y \
                ssh \
                nano && \
    rm -rf /var/lib/apt/lists/*

RUN pip install numpy
RUN pip install sklearn
RUN pip install matplotlib

# Environment to store username, uid and gid.
ENV USER="tavakol"\
    USERID=8013\
    GROUPID=9001\
    GROUPNAME="s876clal"

RUN mkdir -p "/home/${USER}" && \
    echo "${USER}:x:${USERID}:${GROUPID}:${USER}:/home/${USER}:/bin/bash" >> /etc/passwd && \
    echo "${GROUPNAME}:x:${GROUPID}:${USER}" >> /etc/group && \
    chown "${USERID}:${GROUPID}" -R "/home/${USER}"

RUN chown -R "${USERID}:${GROUPID}" "/home/${USER}"

USER "${USER}"

WORKDIR "/home/${USER}"

CMD ["python"]
