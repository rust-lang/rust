//! Tests for user-defined procedural macros.
//!
//! Note `//- proc_macros: identity` fixture metas in tests -- we don't use real
//! proc-macros here, as that would be slow. Instead, we use several hard-coded
//! in-memory macros.
use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]
fn attribute_macro_attr_censoring() {
    cov_mark::check!(attribute_macro_attr_censoring);
    check(
        r#"
//- proc_macros: identity
#[attr1] #[proc_macros::identity] #[attr2]
struct S;
"#,
        expect![[r#"
#[attr1] #[proc_macros::identity] #[attr2]
struct S;

#[attr1]
#[attr2] struct S;"#]],
    );
}

#[test]
fn derive_censoring() {
    cov_mark::check!(derive_censoring);
    check(
        r#"
//- proc_macros: derive_identity
//- minicore:derive
#[attr1]
#[derive(Foo)]
#[derive(proc_macros::DeriveIdentity)]
#[derive(Bar)]
#[attr2]
struct S;
"#,
        expect![[r#"
#[attr1]
#[derive(Foo)]
#[derive(proc_macros::DeriveIdentity)]
#[derive(Bar)]
#[attr2]
struct S;

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
    bar. ;
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

fn#0:1@45..47#0# foo#0:1@48..51#0#(#0:1@51..52#0#&#0:1@52..53#0#self#0:1@53..57#0# )#0:1@57..58#0# {#0:1@59..60#0#
    self#0:1@65..69#0# .#0:1@69..70#0#0#0:1@70..71#0#.#0:1@71..72#0#1#0:1@73..74#0#;#0:1@74..75#0#
}#0:1@76..77#0#"#]],
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
