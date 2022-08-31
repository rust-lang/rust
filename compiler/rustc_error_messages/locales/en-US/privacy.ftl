privacy_field_is_private = field `{$field_name}` of {$variant_descr} `{$def_path_str}` is private
privacy_field_is_private_is_update_syntax_label = field `{$field_name}` is private
privacy_field_is_private_label = private field

privacy_item_is_private = {$kind} `{$descr}` is private
    .label = private {$kind}
privacy_unnamed_item_is_private = {$kind} is private
    .label = private {$kind}

privacy_in_public_interface = {$vis_descr} {$kind} `{$descr}` in public interface
    .label = can't leak {$vis_descr} {$kind}
    .visibility_label = `{$descr}` declared as {$vis_descr}

privacy_from_private_dep_in_public_interface =
    {$kind} `{$descr}` from private dependency '{$krate}' in public interface

privacy_private_in_public_lint =
    {$vis_descr} {$kind} `{$descr}` in public interface (error {$kind ->
        [trait] E0445
        *[other] E0446
    })
