// rustfmt-file_lines: [{"file":"tests/source/file-lines-3.rs","range":[5,9]},{"file":"tests/source/file-lines-3.rs","range":[11,16]}]
// rustfmt-error_on_line_overflow: false

fn floaters() {
    let x = Foo {
                field1: val1,
                field2: val2,
            }
            .method_call().method_call();

    let y = if cond {
                val1
            } else {
                val2
            }
                .method_call();

    {
        match x {
            PushParam => {
                // comment
                stack.push(mparams[match cur.to_digit(10) {
                                            Some(d) => d as usize - 1,
                                            None => return Err("bad param number".to_owned()),
                                        }]
                               .clone());
            }
        }
    }
}
