/bin/bash

set -ex

if [[ -z "$2" ]]; then
  echo "usage: secure-download.sh <URL> <SHA256>"
  exit 1
fi

URL=$1
SHA256=$2
BASENAME=`basename ${URL}`

ON_EXIT="rm -f ${BASENAME}"

trap "rm -f ${BASENAME}" EXIT
curl -L ${URL} > ${BASENAME}
echo "${SHA256} *${BASENAME}" | sha256sum --status -c -
cat ${BASENAME}
rm -f ${BASENAME}
trap - EXIT
