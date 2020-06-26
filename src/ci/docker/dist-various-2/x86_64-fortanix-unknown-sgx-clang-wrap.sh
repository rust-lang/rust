#!/bin/bash

args=("$@")

for i in "${!args[@]}"; do
    # x86_64-fortanix-unknown-sgx doesn't have a C sysroot for things like
    # stdint.h and the C++ STL. Unlike GCC, clang will not use the host's
    # sysroot instead. Force it.
    if [ "${args[$i]}" = "--target=x86_64-fortanix-unknown-sgx" ]; then
        args[$i]="--target=x86_64-unknown-linux-gnu"
    fi
done

exec "${0/x86_64-fortanix-unknown-sgx-clang/clang}" "${args[@]}"
