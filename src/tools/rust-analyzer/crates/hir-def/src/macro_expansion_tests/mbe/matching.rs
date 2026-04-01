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
/* error: expected literal */"#]],
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
    let a = 2 let b = 5 drop(b-a)println!("{}", a+b)
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
| ;
|92| ;
|let x = 92| ;
|loop {}
| ;
|);
"#]],
    );
}

#[test]
fn range_patterns() {
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
stringify!(.. | .. | .. |);
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

// For this test and the one below, see rust-lang/rust#86730.
#[test]
fn expr_dont_match_let_expr() {
    check(
        r#"
macro_rules! foo {
    ($e:expr) => { $e }
}

fn test() {
    foo!(let a = 3);
}
"#,
        expect![[r#"
macro_rules! foo {
    ($e:expr) => { $e }
}

fn test() {
    /* error: no rule matches input tokens */missing;
}
"#]],
    );
}

#[test]
fn expr_inline_const() {
    check(
        r#"
//- /lib.rs edition:2021
macro_rules! foo {
    ($e:expr) => { $e }
}

fn test() {
    foo!(const { 3 });
}
"#,
        expect![[r#"
macro_rules! foo {
    ($e:expr) => { $e }
}

fn test() {
    /* error: no rule matches input tokens */missing;
}
"#]],
    );
    check(
        r#"
//- /lib.rs edition:2024
macro_rules! foo {
    ($e:expr) => { $e }
}

fn test() {
    foo!(const { 3 });
}
"#,
        expect![[r#"
macro_rules! foo {
    ($e:expr) => { $e }
}

fn test() {
    (const {
        3
    }
    );
}
"#]],
    );
}

#[test]
fn meta_variable_raw_name_equals_non_raw() {
    check(
        r#"
macro_rules! m {
    ($r#name:tt) => {
        $name
    }
}

fn test() {
    m!(1234)
}
"#,
        expect![[r#"
macro_rules! m {
    ($r#name:tt) => {
        $name
    }
}

fn test() {
    1234
}
"#]],
    );
}

#[test]
fn meta_fat_arrow() {
    check(
        r#"
macro_rules! m {
    ( $m:meta => ) => {};
}

m! { foo => }
    "#,
        expect![[r#"
macro_rules! m {
    ( $m:meta => ) => {};
}


    "#]],
    );
}
