#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isLinux && [[ $(findmnt -n /) =~ "ext4" ]] ; then
    # noauto_da_alloc since auto_da_alloc causes sync IO for some common file creation patterns
    # lazytime avoids sync IO when (rel)atime updates are applied
    # barrier=0 disables write cache flushing, which can reduce latency at the cost of durability,
    #    but CI machines are ephemeral, so we don't need to be crash-proof.
    #
    # Ideally we'd set additional options, but those would require
    # a reboot or unmounting the rootfs.
    sudo mount -oremount,delalloc,lazytime,barrier=0,noauto_da_alloc /
    sudo bash -c 'echo "write through" > /sys/block/sda/queue/write_cache' || true
    sudo bash -c 'echo "write through" > /sys/block/sdb/queue/write_cache' || true

    ionice -c 3 fstrim / &

    lsblk

    mount
fi
