// rustfmt-match_arm_blocks: false
// rustfmt-control_brace_style: AlwaysNextLine

fn foo() {
    match 0 {
        0 => {
            aaaaaaaaaaaaaaaaaaaaaaaa
                + bbbbbbbbbbbbbbbbbbbbbbbbb
                + bbbbbbbbbbbbbbbbbbbbbbbbb
                + bbbbbbbbbbbbbbbbbbbbbbbbb
                + bbbbbbbbbbbbbbbbbbbbbbbbb
        }
        _ => 2,
    }
}
