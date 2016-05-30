// rustfmt-single_line_if_else_max_width: 0
// rustfmt-chain_base_indent: Inherit
// Test chain formatting with block indented base

fn floaters() {
    let x = Foo {
        field1: val1,
        field2: val2,
    }
    .method_call()
    .method_call();

    let y = if cond {
        val1
    } else {
        val2
    }
    .method_call();

    {
        match x {
            PushParam => {
                // params are 1-indexed
                stack.push(mparams[match cur.to_digit(10) {
                    Some(d) => d as usize - 1,
                    None => return Err("bad param number".to_owned()),
                }]
                .clone());
            }
        }
    }
}
