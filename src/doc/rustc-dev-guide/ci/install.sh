#!/bin/bash
set -ex

if command -v ghp-import >/dev/null 2>&1; then
    pip install ghp-import
fi