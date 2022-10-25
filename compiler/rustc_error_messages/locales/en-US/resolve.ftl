resolve_lang_item_on_incorrect_target =
    `{$name}` language item must be applied to a {$expected_target}
    .label = attribute should be applied to a {$expected_target}, not a {$actual_target}

resolve_unknown_external_lang_item =
    unknown external lang item: `{$lang_item}`

resolve_unknown_lang_item =
    definition of an unknown language item: `{$name}`
    .label = definition of unknown language item `{$name}`

resolve_incorrect_target =
    `{$name}` language item must be applied to a {$kind} with {$at_least ->
        [true] at least {$num}
        *[false] {$num}
    } generic {$num ->
        [one] argument
        *[other] arguments
    }
    .label = this {$kind} has {$actual_num} generic {$actual_num ->
        [one] argument
        *[other] arguments
    }
