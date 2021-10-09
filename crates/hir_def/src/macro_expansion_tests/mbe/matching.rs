//! Test that `$var:expr` captures function correctly.

use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]
fn unary_minus_is_a_literal() {
    check(
        r#"
macro_rules! m { ($x:literal) => (literal!();); ($x:tt) => (not_a_literal!();); }
m!(92);
m!(-92);
m!(-9.2);
m!(--92);
"#,
        expect![[r#"
macro_rules! m { ($x:literal) => (literal!();); ($x:tt) => (not_a_literal!();); }
literal!();
literal!();
literal!();
/* error: leftover tokens */not_a_literal!();
"#]],
    )
}
