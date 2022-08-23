metadata_rlib_required =
    crate `{$crate_name}` required to be available in rlib format, but was not found in this form

metadata_lib_required =
    crate `{$crate_name}` required to be available in {$kind} format, but was not found in this form

metadata_crate_dep_multiple =
    cannot satisfy dependencies so `{$crate_name}` only shows up once
    .help = having upstream crates all available in one format will likely make this go away

metadata_two_panic_runtimes =
    cannot link together two panic runtimes: {$prev_name} and {$cur_name}

metadata_bad_panic_strategy =
    the linked panic runtime `{$runtime}` is not compiled with this crate's panic strategy `{$strategy}`

metadata_required_panic_strategy =
    the crate `{$crate_name}` requires panic strategy `{$found_strategy}` which is incompatible with this crate's strategy of `{$desired_strategy}`

metadata_incompatible_panic_in_drop_strategy =
    the crate `{$crate_name}` is compiled with the panic-in-drop strategy `{$found_strategy}` which is incompatible with this crate's strategy of `{$desired_strategy}`
