// rustfmt-match_arm_indent: false

// Guards are indented if the pattern is longer than 6 characters
fn test() {
    match value {
    LongOption
        if condition || something_else || and_a_third_thing || long_condition || basically =>
    {
        do_stuff();
        other_stuff();
    }

    A23456 if loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong => {
        "1";
        "2";
    }
    }
}

fn complicated() {
    match rewrite {
    // reflows
    Ok(ref body_str)
        if is_block
            || (!body_str.contains('\n') && unicode_str_width(body_str) <= body_shape.width) =>
    {
        return combine_orig_body(body_str);
    }
    _ => rewrite,
    }
}
