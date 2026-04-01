//@ check-fail
//@ revisions: overflow_checks_ debug_assertions_ ub_checks_ sanitize_
//@ revisions: sanitizer_cfi_generalize_pointers_ sanitizer_cfi_normalize_integers_
//@ revisions: proc_macro_ panic_ target_feature_ unix_ windows_ target_abi_
//@ revisions: target_arch_ target_endian_ target_env_ target_family_ target_os_
//@ revisions: target_pointer_width_ target_vendor_ target_has_atomic_
//@ revisions: target_has_atomic_equal_alignment_ target_has_atomic_load_store_
//@ revisions: target_thread_local_ relocation_model_
//@ revisions: fmt_debug_
//@ revisions: emscripten_wasm_eh_
//@ revisions: reliable_f16_ reliable_f16_math_ reliable_f128_ reliable_f128_math_

//@ [overflow_checks_]compile-flags: --cfg overflow_checks
//@ [debug_assertions_]compile-flags: --cfg debug_assertions
//@ [ub_checks_]compile-flags: --cfg ub_checks
//@ [sanitize_]compile-flags: --cfg sanitize="cfi"
//@ [sanitizer_cfi_generalize_pointers_]compile-flags: --cfg sanitizer_cfi_generalize_pointers
//@ [sanitizer_cfi_normalize_integers_]compile-flags: --cfg sanitizer_cfi_normalize_integers
//@ [proc_macro_]compile-flags: --cfg proc_macro
//@ [panic_]compile-flags: --cfg panic="abort"
//@ [target_feature_]compile-flags: --cfg target_feature="sse3"
//@ [unix_]compile-flags: --cfg unix
//@ [windows_]compile-flags: --cfg windows
//@ [target_abi_]compile-flags: --cfg target_abi="gnu"
//@ [target_arch_]compile-flags: --cfg target_arch="arm"
//@ [target_endian_]compile-flags: --cfg target_endian="little"
//@ [target_env_]compile-flags: --cfg target_env
//@ [target_family_]compile-flags: --cfg target_family="unix"
//@ [target_os_]compile-flags: --cfg target_os="linux"
//@ [target_pointer_width_]compile-flags: --cfg target_pointer_width="32"
//@ [target_vendor_]compile-flags: --cfg target_vendor
//@ [target_has_atomic_]compile-flags: --cfg target_has_atomic="32"
//@ [target_has_atomic_equal_alignment_]compile-flags: --cfg target_has_atomic_equal_alignment="32"
//@ [target_has_atomic_load_store_]compile-flags: --cfg target_has_atomic_load_store="32"
//@ [target_thread_local_]compile-flags: --cfg target_thread_local
//@ [relocation_model_]compile-flags: --cfg relocation_model="a"
//@ [fmt_debug_]compile-flags: --cfg fmt_debug="shallow"
//@ [emscripten_wasm_eh_]compile-flags: --cfg emscripten_wasm_eh
//@ [reliable_f16_]compile-flags: --cfg target_has_reliable_f16
//@ [reliable_f16_math_]compile-flags: --cfg target_has_reliable_f16_math
//@ [reliable_f128_]compile-flags: --cfg target_has_reliable_f128
//@ [reliable_f128_math_]compile-flags: --cfg target_has_reliable_f128_math

fn main() {}

//~? ERROR unexpected `--cfg
