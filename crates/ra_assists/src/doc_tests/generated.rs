//! Generated file, do not edit by hand, see `crate/ra_tools/src/codegen`

use super::check;

#[test]
fn doctest_convert_to_guarded_return() {
    check(
        "convert_to_guarded_return",
        r#####"
fn main() {
    <|>if cond {
        foo();
        bar();
    }
}
"#####,
        r#####"
fn main() {
    if !cond {
        return;
    }
    foo();
    bar();
}
"#####,
    )
}
