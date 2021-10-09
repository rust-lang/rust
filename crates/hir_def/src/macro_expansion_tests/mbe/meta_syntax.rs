//! Test for the syntax of macros themselves.

use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]
fn well_formed_macro_rules() {
    check(
        r#"
macro_rules! m {
    ($i:ident) => ();
    ($(x),*) => ();
    ($(x)_*) => ();
    ($(x)i*) => ();
    ($($i:ident)*) => ($_);
    ($($true:ident)*) => ($true);
    ($($false:ident)*) => ($false);
    ($) => (m!($););
}
m!($);
"#,
        expect![[r#"
macro_rules! m {
    ($i:ident) => ();
    ($(x),*) => ();
    ($(x)_*) => ();
    ($(x)i*) => ();
    ($($i:ident)*) => ($_);
    ($($true:ident)*) => ($true);
    ($($false:ident)*) => ($false);
    ($) => (m!($););
}
m!($);
"#]],
    )
}

#[test]
fn malformed_macro_rules() {
    check(
        r#"
macro_rules! i1 { invalid }
i1!();

macro_rules! e1 { $i:ident => () }
e1!();
macro_rules! e2 { ($i:ident) () }
e2!();
macro_rules! e3 { ($(i:ident)_) => () }
e3!();

macro_rules! f1 { ($i) => ($i) }
f1!();
macro_rules! f2 { ($i:) => ($i) }
f2!();
macro_rules! f3 { ($i:_) => () }
f3!();
"#,
        expect![[r#"
macro_rules! i1 { invalid }
/* error: invalid macro definition: expected subtree */

macro_rules! e1 { $i:ident => () }
/* error: invalid macro definition: expected subtree */
macro_rules! e2 { ($i:ident) () }
/* error: invalid macro definition: expected `=` */
macro_rules! e3 { ($(i:ident)_) => () }
/* error: invalid macro definition: invalid repeat */

macro_rules! f1 { ($i) => ($i) }
/* error: invalid macro definition: bad fragment specifier 1 */
macro_rules! f2 { ($i:) => ($i) }
/* error: invalid macro definition: bad fragment specifier 1 */
macro_rules! f3 { ($i:_) => () }
/* error: invalid macro definition: bad fragment specifier 1 */
"#]],
    )
}
