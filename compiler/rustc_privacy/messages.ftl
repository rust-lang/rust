privacy_field_is_private = field `{$field_name}` of {$variant_descr} `{$def_path_str}` is private
privacy_field_is_private_is_update_syntax_label = field `{$field_name}` is private
privacy_field_is_private_label = private field

privacy_from_private_dep_in_public_interface =
    {$kind} `{$descr}` from private dependency '{$krate}' in public interface

privacy_in_public_interface = {$vis_descr} {$kind} `{$descr}` in public interface
    .label = can't leak {$vis_descr} {$kind}
    .visibility_label = `{$descr}` declared as {$vis_descr}

privacy_item_is_private = {$kind} `{$descr}` is private
    .label = private {$kind}
privacy_private_in_public_lint =
    {$vis_descr} {$kind} `{$descr}` in public interface (error {$kind ->
        [trait] E0445
        *[other] E0446
    })

privacy_private_interface_or_bounds_lint = {$ty_kind} `{$ty_descr}` is more private than the item `{$item_descr}`
    .item_note = {$item_kind} `{$item_descr}` is reachable at visibility `{$item_vis_descr}`
    .ty_note = but {$ty_kind} `{$ty_descr}` is only usable at visibility `{$ty_vis_descr}`

privacy_report_effective_visibility = {$descr}

privacy_unnameable_types_lint = {$kind} `{$descr}` is reachable but cannot be named
    .label = reachable at visibility `{$reachable_vis}`, but can only be named at visibility `{$reexported_vis}`

privacy_unnamed_item_is_private = {$kind} is private
    .label = private {$kind}
