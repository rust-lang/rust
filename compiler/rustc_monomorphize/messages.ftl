monomorphize_abi_error_disabled_vector_type =
  this function {$is_call ->
    [true] call
    *[false] definition
  } uses SIMD vector type `{$ty}` which (with the chosen ABI) requires the `{$required_feature}` target feature, which is not enabled{$is_call ->
    [true] {" "}in the caller
    *[false] {""}
  }
  .label = function {$is_call ->
    [true] called
    *[false] defined
  } here
  .help = consider enabling it globally (`-C target-feature=+{$required_feature}`) or locally (`#[target_feature(enable="{$required_feature}")]`)

monomorphize_abi_error_unsupported_vector_type =
  this function {$is_call ->
    [true] call
    *[false] definition
  } uses SIMD vector type `{$ty}` which is not currently supported with the chosen ABI
  .label = function {$is_call ->
    [true] called
    *[false] defined
  } here

monomorphize_abi_required_target_feature =
  this function {$is_call ->
    [true] call
    *[false] definition
  } uses ABI "{$abi}" which requires the `{$required_feature}` target feature, which is not enabled{$is_call ->
    [true] {" "}in the caller
    *[false] {""}
  }
  .label = function {$is_call ->
    [true] called
    *[false] defined
  } here
  .help = consider enabling it globally (`-C target-feature=+{$required_feature}`) or locally (`#[target_feature(enable="{$required_feature}")]`)

monomorphize_couldnt_dump_mono_stats =
    unexpected error occurred while dumping monomorphization stats: {$error}

monomorphize_encountered_error_while_instantiating =
    the above error was encountered while instantiating `{$formatted_item}`

monomorphize_large_assignments =
    moving {$size} bytes
    .label = value moved from here
    .note = The current maximum size is {$limit}, but it can be customized with the move_size_limit attribute: `#![move_size_limit = "..."]`

monomorphize_no_optimized_mir =
    missing optimized MIR for `{$instance}` in the crate `{$crate_name}`
    .note = missing optimized MIR for this item (was the crate `{$crate_name}` compiled with `--emit=metadata`?)

monomorphize_recursion_limit =
    reached the recursion limit while instantiating `{$shrunk}`
    .note = `{$def_path_str}` defined here

monomorphize_start_not_found = using `fn main` requires the standard library
    .help = use `#![no_main]` to bypass the Rust generated entrypoint and declare a platform specific entrypoint yourself, usually with `#[no_mangle]`

monomorphize_symbol_already_defined = symbol `{$symbol}` is already defined

monomorphize_written_to_path = the full type name has been written to '{$path}'
