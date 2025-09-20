#[test]
fn quote_command_line_args() {
    use super::quote_command_line_args;

    struct Case<'a> {
        args: &'a [&'a str],
        expected: &'a str,
    }

    let cases = &[
        Case { args: &[], expected: "" },
        Case { args: &["--hello", "world"], expected: r#""--hello" "world""# },
        Case { args: &["--hello world"], expected: r#""--hello world""# },
        Case {
            args: &["plain", "$dollar", "spa ce", r"back\slash", r#""quote""#, "plain"],
            expected: r#""plain" "\$dollar" "spa ce" "back\\slash" "\"quote\"" "plain""#,
        },
    ];

    for &Case { args, expected } in cases {
        let args = args.iter().copied().map(str::to_owned).collect::<Vec<_>>();
        let actual = quote_command_line_args(&args);
        assert_eq!(actual, expected, "args {args:?}");
    }
}
