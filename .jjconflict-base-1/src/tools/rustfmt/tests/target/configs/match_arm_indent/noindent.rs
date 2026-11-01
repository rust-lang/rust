// rustfmt-match_arm_indent: false
// Don't indent the match arms

fn foo() {
    match value {
    0 => {
        "one";
        "two";
    }
    1 | 2 | 3 => {
        "line1";
        "line2";
    }
    100..1000 => oneline(),

    _ => {
        // catch-all
        todo!();
    }
    }
}
