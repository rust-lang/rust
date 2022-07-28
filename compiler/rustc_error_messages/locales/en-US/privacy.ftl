privacy-field-is-private = field `{$field_name}` of {$variant_descr} `{$def_path_str}` is private
privacy-field-is-private-is-update-syntax-label = field `{$field_name}` is private
privacy-field-is-private-label = private field

privacy-item-is-private = {$kind} `{$descr}` is private
    .label = private {$kind}
privacy-unnamed-item-is-private = {$kind} is private
    .label = private {$kind}

privacy-in-public-interface = {$vis_descr} {$kind} `{$descr}` in public interface
    .label = can't leak {$vis_descr} {$kind}
    .visibility-label = `{$descr}` declared as {$vis_descr}

privacy-from-private-dep-in-public-interface =
    {$kind} `{$descr}` from private dependency '{$krate}' in public interface

private-in-public-lint =
    {$vis_descr} {$kind} `{$descr}` in public interface (error {$kind ->
        [trait] E0445
        *[other] E0446
    })
