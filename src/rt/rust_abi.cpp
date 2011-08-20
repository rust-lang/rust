#include <cstdlib>
#include <stdint.h>
#include "rust_abi.h"

weak_symbol<uint32_t> abi_version("rust_abi_version");

uint32_t get_abi_version() {
    return (*abi_version == NULL) ? 0 : **abi_version;
}

