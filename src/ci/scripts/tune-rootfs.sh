#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isLinux && [[ $(findmnt -n /) =~ "ext4" ]] ; then
    # noauto_da_alloc since auto_da_alloc causes sync IO for some common file creation patters
    # lazytime avoids sync IO when (rel)atime updates are applied
    # nodiscard because the ext4 doesn't support async discard (unlike xfs or btrfs)
    sudo mount -oremount,lazytime,nodiscard,noauto_da_alloc /

    mount
fi
