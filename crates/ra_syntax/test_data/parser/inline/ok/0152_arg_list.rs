fn assert_float(s: &str, n: f64) {}
fn foo() {
    assert_float(
        "1.797693134862315708e+308L",
        #[allow(clippy::excessive_precision)]
        #[allow(dead_code)]
        1.797_693_134_862_315_730_8e+308,
    );
}
