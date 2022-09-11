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

#[test]
fn test_expand_bad_literal() {
    check(
        r#"
macro_rules! m { ($i:literal) => {}; }
m!(&k");
"#,
        expect![[r#"
macro_rules! m { ($i:literal) => {}; }
/* error: Failed to lower macro args to token tree */"#]],
    );
}

#[test]
fn test_empty_comments() {
    check(
        r#"
macro_rules! m{ ($fmt:expr) => (); }
m!(/**/);
"#,
        expect![[r#"
macro_rules! m{ ($fmt:expr) => (); }
/* error: expected Expr */
"#]],
    );
}

#[test]
fn asi() {
    // Thanks, Christopher!
    //
    // https://internals.rust-lang.org/t/understanding-decisions-behind-semicolons/15181/29
    check(
        r#"
macro_rules! asi { ($($stmt:stmt)*) => ($($stmt)*); }

fn main() {
    asi! {
        let a = 2
        let b = 5
        drop(b-a)
        println!("{}", a+b)
    }
}
"#,
        expect![[r#"
macro_rules! asi { ($($stmt:stmt)*) => ($($stmt)*); }

fn main() {
    let a = 2let b = 5drop(b-a)println!("{}", a+b)
}
"#]],
    )
}

#[test]
fn stmt_boundaries() {
    // FIXME: this actually works OK under rustc.
    check(
        r#"
macro_rules! m {
    ($($s:stmt)*) => (stringify!($($s |)*);)
}
m!(;;92;let x = 92; loop {};);
"#,
        expect![[r#"
macro_rules! m {
    ($($s:stmt)*) => (stringify!($($s |)*);)
}
stringify!(;
|;
|92|;
|let x = 92|;
|loop {}
|;
|);
"#]],
    );
}

#[test]
fn range_patterns() {
    // FIXME: rustc thinks there are three patterns here, not one.
    check(
        r#"
macro_rules! m {
    ($($p:pat)*) => (stringify!($($p |)*);)
}
m!(.. .. ..);
"#,
        expect![[r#"
macro_rules! m {
    ($($p:pat)*) => (stringify!($($p |)*);)
}
stringify!(.. .. ..|);
"#]],
    );
}

#[test]
fn trailing_vis() {
    check(
        r#"
macro_rules! m { ($($i:ident)? $vis:vis) => () }
m!(x pub);
"#,
        expect![[r#"
macro_rules! m { ($($i:ident)? $vis:vis) => () }

"#]],
    )
}
