monomorphize_couldnt_dump_mono_stats =
    unexpected error occurred while dumping monomorphization stats: {$error}

monomorphize_encountered_error_while_instantiating =
    the above error was encountered while instantiating `{$formatted_item}`

monomorphize_no_optimized_mir =
    missing optimized MIR for `{$instance}` in the crate `{$crate_name}`
    .note = missing optimized MIR for this item (was the crate `{$crate_name}` compiled with `--emit=metadata`?)

monomorphize_recursion_limit =
    reached the recursion limit while instantiating `{$shrunk}`
    .note = `{$def_path_str}` defined here

monomorphize_start_not_found = using `fn main` requires the standard library
    .help = use `#![no_main]` to bypass the Rust generated entrypoint and declare a platform specific entrypoint yourself, usually with `#[no_mangle]`

monomorphize_symbol_already_defined = symbol `{$symbol}` is already defined

monomorphize_wasm_c_abi_transition =
    this function {$is_call ->
      [true] call
      *[false] definition
    } involves an argument of type `{$ty}` which is affected by the wasm ABI transition
    .help = the "C" ABI Rust uses on wasm32-unknown-unknown will change to align with the standard "C" ABI for this target

monomorphize_written_to_path = the full type name has been written to '{$path}'
