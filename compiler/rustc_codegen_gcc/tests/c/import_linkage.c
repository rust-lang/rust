/* The symbols that `tests/run/import_linkage.rs` imports with an explicit `#[linkage]`.
 *
 * Such an import is a pointer whose value is the address of the symbol, so what the Rust side
 * reads back is `&value_*`, not the pointer stored in it. The distinct values make a mix-up
 * visible. */

#include <stdint.h>

int32_t external_value = 1;
int32_t available_externally_value = 2;
int32_t linkonce_value = 3;
int32_t linkonce_odr_value = 4;
int32_t weak_value = 5;
int32_t weak_odr_value = 6;
int32_t common_value = 7;
int32_t extern_weak_value = 8;
