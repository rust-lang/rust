#!/bin/bash

set -eux

python3 ../x.py dist \
    --host $HOSTS --target $HOSTS \
    --include-default-paths \
    build-manifest bootstrap
