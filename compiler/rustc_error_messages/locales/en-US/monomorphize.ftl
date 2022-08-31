monomorphize_recursion_limit =
    reached the recursion limit while instantiating `{$shrunk}`
    .note = `{$def_path_str}` defined here

monomorphize_written_to_path = the full type name has been written to '{$path}'

monomorphize_type_length_limit = reached the type-length limit while instantiating `{$shrunk}`

monomorphize_consider_type_length_limit =
    consider adding a `#![type_length_limit="{$type_length}"]` attribute to your crate

monomorphize_fatal_error = {$error_message}

monomorphize_unknown_partition_strategy = unknown partitioning strategy

monomorphize_symbol_already_defined = symbol `{$symbol}` is already defined

monomorphize_unused_generic_params = item has unused generic parameters

monomorphize_large_assignments =
    moving {$size} bytes
    .label = value moved from here
    .note = The current maximum size is {$limit}, but it can be customized with the move_size_limit attribute: `#![move_size_limit = "..."]`

monomorphize_requires_lang_item =
    requires `{$lang_item}` lang_item
