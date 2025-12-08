//! Tests for user-defined procedural macros.
//!
//! Note `//- proc_macros: identity` fixture metas in tests -- we don't use real
//! proc-macros here, as that would be slow. Instead, we use several hard-coded
//! in-memory macros.
use expect_test::expect;

use crate::macro_expansion_tests::{check, check_errors};

#[test]
fn attribute_macro_attr_censoring() {
    check(
        r#"
//- proc_macros: identity
//- minicore: derive
#[attr1] #[derive()] #[proc_macros::identity] #[attr2]
struct S;

/// Foo
#[cfg_attr(false, doc = "abc...", attr1)]
mod foo {
    #![cfg_attr(true, cfg_attr(true, foo, cfg_attr(false, bar), proc_macros::identity))]
    #![cfg_attr(true, doc = "123...", attr2)]
    #![attr3]

    #[cfg_attr(true, cfg(false))]
    fn foo() {}

    #[cfg(true)]
    fn bar() {}
}
"#,
        expect![[r##"
#[attr1] #[derive()] #[proc_macros::identity] #[attr2]
struct S;

/// Foo
#[cfg_attr(false, doc = "abc...", attr1)]
mod foo {
    #![cfg_attr(true, cfg_attr(true, foo, cfg_attr(false, bar), proc_macros::identity))]
    #![cfg_attr(true, doc = "123...", attr2)]
    #![attr3]

    #[cfg_attr(true, cfg(false))]
    fn foo() {}

    #[cfg(true)]
    fn bar() {}
}

#[attr1]
#[attr2] struct S;
#[doc = " Foo"] mod foo {
    # ![foo]
    # ![doc = "123..."]
    # ![attr2]
    # ![attr3]
    #[cfg_attr(true , cfg(false ))] fn foo() {}
    #[cfg(true )] fn bar() {}
}"##]],
    );
}

#[test]
fn derive_censoring() {
    check(
        r#"
//- proc_macros: derive_identity
//- minicore:derive
use derive as my_cool_derive;
#[attr1]
#[derive(Foo)]
#[derive(proc_macros::DeriveIdentity)]
#[derive(Bar)]
#[attr2]
struct S;

#[my_cool_derive()]
#[cfg_attr(true, derive(), attr1, derive(proc_macros::DeriveIdentity))]
#[my_cool_derive()]
struct Foo {
    #[cfg_attr(false, cfg(false), attr2)]
    v1: i32,
    #[cfg_attr(true, cfg(false), attr2)]
    v1: i32,
    #[cfg_attr(true, attr3)]
    v2: fn(#[cfg(false)] param: i32, #[cfg_attr(true, attr4)] param2: u32),
    v3: Foo<{
        #[cfg(false)]
        let foo = 123;
        456
    }>,
    #[cfg(false)]
    v4: bool // No comma here
}
"#,
        expect![[r#"
use derive as my_cool_derive;
#[attr1]
#[derive(Foo)]
#[derive(proc_macros::DeriveIdentity)]
#[derive(Bar)]
#[attr2]
struct S;

#[my_cool_derive()]
#[cfg_attr(true, derive(), attr1, derive(proc_macros::DeriveIdentity))]
#[my_cool_derive()]
struct Foo {
    #[cfg_attr(false, cfg(false), attr2)]
    v1: i32,
    #[cfg_attr(true, cfg(false), attr2)]
    v1: i32,
    #[cfg_attr(true, attr3)]
    v2: fn(#[cfg(false)] param: i32, #[cfg_attr(true, attr4)] param2: u32),
    v3: Foo<{
        #[cfg(false)]
        let foo = 123;
        456
    }>,
    #[cfg(false)]
    v4: bool // No comma here
}

#[attr1]
#[my_cool_derive()] struct Foo {
    v1: i32, #[attr3]v2: fn(#[attr4]param2: u32), v3: Foo< {
        456
    }
    >,
}
#[attr1]
#[derive(Bar)]
#[attr2] struct S;"#]],
    );
}

#[test]
fn attribute_macro_syntax_completion_1() {
    // this is just the case where the input is actually valid
    check(
        r#"
//- proc_macros: identity_when_valid
#[proc_macros::identity_when_valid]
fn foo() { bar.baz(); blub }
"#,
        expect![[r#"
#[proc_macros::identity_when_valid]
fn foo() { bar.baz(); blub }

fn foo() {
    bar.baz();
    blub
}"#]],
    );
}

#[test]
fn attribute_macro_syntax_completion_2() {
    // common case of dot completion while typing
    check(
        r#"
//- proc_macros: identity_when_valid
#[proc_macros::identity_when_valid]
fn foo() { bar.; blub }
"#,
        expect![[r#"
#[proc_macros::identity_when_valid]
fn foo() { bar.; blub }

fn foo() {
    bar.;
    blub
}"#]],
    );
}

#[test]
fn macro_rules_in_attr() {
    // Regression test for https://github.com/rust-lang/rust-analyzer/issues/12211
    check(
        r#"
//- proc_macros: identity
macro_rules! id {
    ($($t:tt)*) => {
        $($t)*
    };
}
id! {
    #[proc_macros::identity]
    impl Foo for WrapBj {
        async fn foo(&self) {
            self.id().await;
        }
    }
}
"#,
        expect![[r#"
macro_rules! id {
    ($($t:tt)*) => {
        $($t)*
    };
}
#[proc_macros::identity] impl Foo for WrapBj {
    async fn foo(&self ) {
        self .id().await ;
    }
}
"#]],
    );
}

#[test]
fn float_parsing_panic() {
    // Regression test for https://github.com/rust-lang/rust-analyzer/issues/12211
    check(
        r#"
//- proc_macros: identity
macro_rules! id {
    ($($t:tt)*) => {
        $($t)*
    };
}
id! {
    #[proc_macros::identity]
    impl Foo for WrapBj {
        async fn foo(&self) {
            self.0. id().await;
        }
    }
}
"#,
        expect![[r#"
macro_rules! id {
    ($($t:tt)*) => {
        $($t)*
    };
}
#[proc_macros::identity] impl Foo for WrapBj {
    async fn foo(&self ) {
        self .0.id().await ;
    }
}
"#]],
    );
}

#[test]
fn float_attribute_mapping() {
    check(
        r#"
//- proc_macros: identity
//+spans+syntaxctxt
#[proc_macros::identity]
fn foo(&self) {
    self.0. 1;
}
"#,
        expect![[r#"
//+spans+syntaxctxt
#[proc_macros::identity]
fn foo(&self) {
    self.0. 1;
}

fn#0:Fn[8A31, 0]@45..47#ROOT2024# foo#0:Fn[8A31, 0]@48..51#ROOT2024#(#0:Fn[8A31, 0]@51..52#ROOT2024#&#0:Fn[8A31, 0]@52..53#ROOT2024#self#0:Fn[8A31, 0]@53..57#ROOT2024# )#0:Fn[8A31, 0]@57..58#ROOT2024# {#0:Fn[8A31, 0]@59..60#ROOT2024#
    self#0:Fn[8A31, 0]@65..69#ROOT2024# .#0:Fn[8A31, 0]@69..70#ROOT2024#0#0:Fn[8A31, 0]@70..71#ROOT2024#.#0:Fn[8A31, 0]@71..72#ROOT2024#1#0:Fn[8A31, 0]@73..74#ROOT2024#;#0:Fn[8A31, 0]@74..75#ROOT2024#
}#0:Fn[8A31, 0]@76..77#ROOT2024#"#]],
    );
}

#[test]
fn attribute_macro_doc_desugaring() {
    check(
        r#"
//- proc_macros: identity
#[proc_macros::identity]
/// doc string \n with newline
/**
     MultiLines Doc
     MultiLines Doc
*/
#[doc = "doc attr"]
struct S;
"#,
        expect![[r##"
#[proc_macros::identity]
/// doc string \n with newline
/**
     MultiLines Doc
     MultiLines Doc
*/
#[doc = "doc attr"]
struct S;

#[doc = " doc string \\n with newline"]
#[doc = "\n     MultiLines Doc\n     MultiLines Doc\n"]
#[doc = "doc attr"] struct S;"##]],
    );
}

#[test]
fn cfg_evaluated_before_attr_macros() {
    check_errors(
        r#"
//- proc_macros: disallow_cfg

use proc_macros::disallow_cfg;

#[disallow_cfg] #[cfg(false)] fn foo() {}
// True cfg are kept.
// #[disallow_cfg] #[cfg(true)] fn bar() {}
#[disallow_cfg] #[cfg_attr(false, inline)] fn baz() {}
#[disallow_cfg] #[cfg_attr(true, inline)] fn qux() {}
    "#,
        expect![[r#""#]],
    );
}

#[test]
fn derive_helpers_are_ignored() {
    check(
        r#"
//- proc_macros: identity, helper_should_be_ignored, helper_should_be_ignored_derive
//- minicore: derive
use proc_macros::{identity, helper_should_be_ignored, HelperShouldBeIgnoredDerive};

#[derive(HelperShouldBeIgnoredDerive)]
#[helper_should_be_ignored]
#[identity]
struct Foo;
"#,
        expect![[r#"
use proc_macros::{identity, helper_should_be_ignored, HelperShouldBeIgnoredDerive};

#[derive(HelperShouldBeIgnoredDerive)]
#[helper_should_be_ignored]
#[identity]
struct Foo;

#[helper_should_be_ignored] struct Foo;"#]],
    );
}

#[test]
fn attribute_macro_stripping_with_cfg() {
    check(
        r#"
//- proc_macros: generate_suffixed_type
#[cfg(all())]
#[proc_macros::generate_suffixed_type]
struct S;
"#,
        expect![[r#"
#[cfg(all())]
#[proc_macros::generate_suffixed_type]
struct S;

struct S;
struct SSuffix;"#]],
    );
}
