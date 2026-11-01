fn foo() {
    let with_alignment = if condition__uses_alignment_for_first_if__0
        || condition__uses_alignment_for_first_if__1
        || condition__uses_alignment_for_first_if__2
    {
    } else if condition__no_alignment_for_later_else__0
        || condition__no_alignment_for_later_else__1
        || condition__no_alignment_for_later_else__2
    {
    };
}
