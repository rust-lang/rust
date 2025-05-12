#!/bin/sh
set -ex

groupadd -r rustbuild && useradd -m -r -g rustbuild rustbuild
mkdir /x-tools && chown rustbuild:rustbuild /x-tools
