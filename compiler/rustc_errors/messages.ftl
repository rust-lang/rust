errors_delayed_at_with_newline =
    delayed at {$emitted_at}
    {$note}

errors_delayed_at_without_newline =
    delayed at {$emitted_at} - {$note}

errors_expected_lifetime_parameter =
    expected lifetime {$count ->
        [1] parameter
        *[other] parameters
    }

errors_indicate_anonymous_lifetime =
    indicate the anonymous {$count ->
        [1] lifetime
        *[other] lifetimes
    }

errors_invalid_flushed_delayed_diagnostic_level =
    `flushed_delayed` got diagnostic with level {$level}, instead of the expected `DelayedBug`

errors_target_inconsistent_architecture =
    inconsistent target specification: "data-layout" claims architecture is {$dl}-endian, while "target-endian" is `{$target}`

errors_target_inconsistent_pointer_width =
    inconsistent target specification: "data-layout" claims pointers are {$pointer_size}-bit, while "target-pointer-width" is `{$target}`

errors_target_invalid_address_space =
    invalid address space `{$addr_space}` for `{$cause}` in "data-layout": {$err}

errors_target_invalid_alignment =
    invalid alignment for `{$cause}` in "data-layout": `{$align}` is {$err_kind ->
        [not_power_of_two] not a power of 2
        [too_large] too large
        *[other] {""}
    }

errors_target_invalid_bits =
    invalid {$kind} `{$bit}` for `{$cause}` in "data-layout": {$err}

errors_target_invalid_bits_size = {$err}

errors_target_invalid_datalayout_pointer_spec =
    unknown pointer specification `{$err}` in datalayout string

errors_target_missing_alignment =
    missing alignment for `{$cause}` in "data-layout"
