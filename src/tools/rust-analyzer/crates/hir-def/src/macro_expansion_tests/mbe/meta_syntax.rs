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
    ($(x)'a*) => ();
    ($(x)'_*) => ();
    ($($i:ident)*) => ($_);
    ($($true:ident)*) => ($true);
    ($($false:ident)*) => ($false);
    (double_dollar) => ($$);
    ($) => (m!($););
    ($($t:tt)*) => ($( ${ignore($t)} ${index()} )-*);
}
m!($);
"#,
        expect![[r#"
macro_rules! m {
    ($i:ident) => ();
    ($(x),*) => ();
    ($(x)_*) => ();
    ($(x)i*) => ();
    ($(x)'a*) => ();
    ($(x)'_*) => ();
    ($($i:ident)*) => ($_);
    ($($true:ident)*) => ($true);
    ($($false:ident)*) => ($false);
    (double_dollar) => ($$);
    ($) => (m!($););
    ($($t:tt)*) => ($( ${ignore($t)} ${index()} )-*);
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

macro_rules! m1 { ($$i) => () }
m1!();
macro_rules! m2 { () => ( ${invalid()} ) }
m2!();
"#,
        expect![[r#"
macro_rules! i1 { invalid }
/* error: macro definition has parse errors */

macro_rules! e1 { $i:ident => () }
/* error: macro definition has parse errors */
macro_rules! e2 { ($i:ident) () }
/* error: macro definition has parse errors */
macro_rules! e3 { ($(i:ident)_) => () }
/* error: macro definition has parse errors */

macro_rules! f1 { ($i) => ($i) }
/* error: macro definition has parse errors */
macro_rules! f2 { ($i:) => ($i) }
/* error: macro definition has parse errors */
macro_rules! f3 { ($i:_) => () }
/* error: macro definition has parse errors */

macro_rules! m1 { ($$i) => () }
/* error: macro definition has parse errors */
macro_rules! m2 { () => ( ${invalid()} ) }
/* error: macro definition has parse errors */
"#]],
    )
}

#[test]
fn test_rustc_issue_57597() {
    // <https://github.com/rust-lang/rust/blob/master/tests/ui/issues/issue-57597.rs>
    check(
        r#"
macro_rules! m0 { ($($($i:ident)?)+) => {}; }
macro_rules! m1 { ($($($i:ident)?)*) => {}; }
macro_rules! m2 { ($($($i:ident)?)?) => {}; }
macro_rules! m3 { ($($($($i:ident)?)?)?) => {}; }
macro_rules! m4 { ($($($($i:ident)*)?)?) => {}; }
macro_rules! m5 { ($($($($i:ident)?)*)?) => {}; }
macro_rules! m6 { ($($($($i:ident)?)?)*) => {}; }
macro_rules! m7 { ($($($($i:ident)*)*)?) => {}; }
macro_rules! m8 { ($($($($i:ident)?)*)*) => {}; }
macro_rules! m9 { ($($($($i:ident)?)*)+) => {}; }
macro_rules! mA { ($($($($i:ident)+)?)*) => {}; }
macro_rules! mB { ($($($($i:ident)+)*)?) => {}; }

m0!();
m1!();
m2!();
m3!();
m4!();
m5!();
m6!();
m7!();
m8!();
m9!();
mA!();
mB!();
    "#,
        expect![[r#"
macro_rules! m0 { ($($($i:ident)?)+) => {}; }
macro_rules! m1 { ($($($i:ident)?)*) => {}; }
macro_rules! m2 { ($($($i:ident)?)?) => {}; }
macro_rules! m3 { ($($($($i:ident)?)?)?) => {}; }
macro_rules! m4 { ($($($($i:ident)*)?)?) => {}; }
macro_rules! m5 { ($($($($i:ident)?)*)?) => {}; }
macro_rules! m6 { ($($($($i:ident)?)?)*) => {}; }
macro_rules! m7 { ($($($($i:ident)*)*)?) => {}; }
macro_rules! m8 { ($($($($i:ident)?)*)*) => {}; }
macro_rules! m9 { ($($($($i:ident)?)*)+) => {}; }
macro_rules! mA { ($($($($i:ident)+)?)*) => {}; }
macro_rules! mB { ($($($($i:ident)+)*)?) => {}; }

/* error: macro definition has parse errors */
/* error: macro definition has parse errors */
/* error: macro definition has parse errors */
/* error: macro definition has parse errors */
/* error: macro definition has parse errors */
/* error: macro definition has parse errors */
/* error: macro definition has parse errors */
/* error: macro definition has parse errors */
/* error: macro definition has parse errors */
/* error: macro definition has parse errors */
/* error: macro definition has parse errors */
/* error: macro definition has parse errors */
    "#]],
    );
}
