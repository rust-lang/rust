//! Completion tests for attributes.
use expect_test::expect;

use crate::tests::{check, check_edit};

#[test]
fn derive_helpers() {
    check(
        r#"
//- /mac.rs crate:mac
#![crate_type = "proc-macro"]

#[proc_macro_derive(MyDerive, attributes(my_cool_helper_attribute))]
pub fn my_derive() {}

//- /lib.rs crate:lib deps:mac
#[rustc_builtin_macro]
pub macro derive($item:item) {}

#[derive(mac::MyDerive)]
pub struct Foo(#[m$0] i32);
"#,
        expect![[r#"
            at allow(…)
            at automatically_derived
            at cfg(…)
            at cfg_attr(…)
            at cold
            at deny(…)
            at deprecated
            at derive                                  macro derive
            at derive(…)
            at diagnostic::do_not_recommend
            at diagnostic::on_unimplemented
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at export_name = "…"
            at forbid(…)
            at global_allocator
            at ignore = "…"
            at inline
            at link
            at link_name = "…"
            at link_section = "…"
            at macro_export
            at macro_use
            at must_use
            at my_cool_helper_attribute derive helper of `MyDerive`
            at no_mangle
            at non_exhaustive
            at panic_handler
            at path = "…"
            at proc_macro
            at proc_macro_attribute
            at proc_macro_derive(…)
            at repr(…)
            at should_panic
            at target_feature(enable = "…")
            at test
            at track_caller
            at used
            at warn(…)
            md mac
            kw crate::
            kw self::
        "#]],
    )
}

#[test]
fn proc_macros() {
    check(
        r#"
//- proc_macros: identity
#[$0]
struct Foo;
"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at derive(…)
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at non_exhaustive
            at repr(…)
            at warn(…)
            md proc_macros
            kw crate::
            kw self::
        "#]],
    )
}

#[test]
fn proc_macros_on_comment() {
    check(
        r#"
//- proc_macros: identity
/// $0
#[proc_macros::identity]
struct Foo;
"#,
        expect![[r#""#]],
    )
}

#[test]
fn proc_macros_qualified() {
    check(
        r#"
//- proc_macros: identity
#[proc_macros::$0]
struct Foo;
"#,
        expect![[r#"
            at identity proc_macro identity
        "#]],
    )
}

#[test]
fn with_existing_attr() {
    check(
        r#"#[no_mangle] #[$0] mcall!();"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at expect(…)
            at forbid(…)
            at warn(…)
            kw crate::
            kw self::
        "#]],
    )
}

#[test]
fn attr_on_source_file() {
    check(
        r#"#![$0]"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at crate_name = ""
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at feature(…)
            at forbid(…)
            at must_use
            at no_implicit_prelude
            at no_main
            at no_mangle
            at no_std
            at recursion_limit = "…"
            at type_length_limit = …
            at warn(…)
            at windows_subsystem = "…"
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_on_module() {
    check(
        r#"#[$0] mod foo;"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at macro_use
            at must_use
            at no_mangle
            at path = "…"
            at warn(…)
            kw crate::
            kw self::
            kw super::
        "#]],
    );
    check(
        r#"mod foo {#![$0]}"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_implicit_prelude
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
            kw super::
        "#]],
    );
}

#[test]
fn attr_on_macro_rules() {
    check(
        r#"#[$0] macro_rules! foo {}"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at macro_export
            at macro_use
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_on_macro_def() {
    check(
        r#"#[$0] macro foo {}"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_on_extern_crate() {
    check(
        r#"#[$0] extern crate foo;"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at macro_use
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_on_use() {
    check(
        r#"#[$0] use foo;"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_on_type_alias() {
    check(
        r#"#[$0] type foo = ();"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_on_struct() {
    check(
        r#"
//- minicore:derive
#[$0]
struct Foo;
"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at derive             macro derive
            at derive(…)
            at derive_const macro derive_const
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at non_exhaustive
            at repr(…)
            at warn(…)
            md core
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_on_enum() {
    check(
        r#"#[$0] enum Foo {}"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at derive(…)
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at non_exhaustive
            at repr(…)
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_on_const() {
    check(
        r#"#[$0] const FOO: () = ();"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_on_static() {
    check(
        r#"#[$0] static FOO: () = ()"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at export_name = "…"
            at forbid(…)
            at global_allocator
            at link_name = "…"
            at link_section = "…"
            at must_use
            at no_mangle
            at used
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_on_trait() {
    check(
        r#"#[$0] trait Foo {}"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at diagnostic::on_unimplemented
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_on_impl() {
    check(
        r#"#[$0] impl () {}"#,
        expect![[r#"
            at allow(…)
            at automatically_derived
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at diagnostic::do_not_recommend
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
    check(
        r#"impl () {#![$0]}"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_with_qualifier() {
    check(
        r#"#[diagnostic::$0] impl () {}"#,
        expect![[r#"
            at allow(…)
            at automatically_derived
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at do_not_recommend
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at warn(…)
        "#]],
    );
    check(
        r#"#[diagnostic::$0] trait Foo {}"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at on_unimplemented
            at warn(…)
        "#]],
    );
}

#[test]
fn attr_diagnostic_on_unimplemented() {
    check(
        r#"#[diagnostic::on_unimplemented($0)] trait Foo {}"#,
        expect![[r#"
            ba label = "…"
            ba message = "…"
            ba note = "…"
        "#]],
    );
    check(
        r#"#[diagnostic::on_unimplemented(message = "foo", $0)] trait Foo {}"#,
        expect![[r#"
            ba label = "…"
            ba note = "…"
        "#]],
    );
    check(
        r#"#[diagnostic::on_unimplemented(note = "foo", $0)] trait Foo {}"#,
        expect![[r#"
            ba label = "…"
            ba message = "…"
            ba note = "…"
        "#]],
    );
}

#[test]
fn attr_on_extern_block() {
    check(
        r#"#[$0] extern {}"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at link
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
    check(
        r#"extern {#![$0]}"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at link
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_on_variant() {
    check(
        r#"enum Foo { #[$0] Bar }"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at expect(…)
            at forbid(…)
            at non_exhaustive
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_on_fn() {
    check(
        r#"#[$0] fn main() {}"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at cold
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at export_name = "…"
            at forbid(…)
            at ignore = "…"
            at inline
            at link_name = "…"
            at link_section = "…"
            at must_use
            at no_mangle
            at panic_handler
            at proc_macro
            at proc_macro_attribute
            at proc_macro_derive(…)
            at should_panic
            at target_feature(enable = "…")
            at test
            at track_caller
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn attr_in_source_file_end() {
    check(
        r#"#[$0]"#,
        expect![[r#"
            at allow(…)
            at automatically_derived
            at cfg(…)
            at cfg_attr(…)
            at cold
            at deny(…)
            at deprecated
            at derive(…)
            at diagnostic::do_not_recommend
            at diagnostic::on_unimplemented
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at export_name = "…"
            at forbid(…)
            at global_allocator
            at ignore = "…"
            at inline
            at link
            at link_name = "…"
            at link_section = "…"
            at macro_export
            at macro_use
            at must_use
            at no_mangle
            at non_exhaustive
            at panic_handler
            at path = "…"
            at proc_macro
            at proc_macro_attribute
            at proc_macro_derive(…)
            at repr(…)
            at should_panic
            at target_feature(enable = "…")
            at test
            at track_caller
            at used
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn invalid_path() {
    check(
        r#"
//- proc_macros: identity
#[proc_macros:::$0]
struct Foo;
"#,
        expect![[r#""#]],
    );

    check(
        r#"
//- minicore: derive, copy
mod foo {
    pub use Copy as Bar;
}
#[derive(foo:::::$0)]
struct Foo;
"#,
        expect![""],
    );
}

#[test]
fn issue_17479() {
    check(
        r#"
//- proc_macros: issue_17479
fn main() {
    proc_macros::issue_17479!("te$0");
}
"#,
        expect![""],
    );
    check(
        r#"
//- proc_macros: issue_17479
fn main() {
    proc_macros::issue_17479!("$0");
}
"#,
        expect![""],
    )
}

mod cfg {
    use super::*;

    #[test]
    fn inside_cfg() {
        check(
            r#"
//- /main.rs cfg:test,dbg=false,opt_level=2
#[cfg($0)]
"#,
            expect![[r#"
                ba dbg
                ba opt_level
                ba test
                ba true
            "#]],
        );
        check(
            r#"
//- /main.rs cfg:test,dbg=false,opt_level=2
#[cfg(b$0)]
"#,
            expect![[r#"
                ba dbg
                ba opt_level
                ba test
                ba true
            "#]],
        );
    }

    #[test]
    fn cfg_target_endian() {
        check(
            r#"#[cfg(target_endian = $0"#,
            expect![[r#"
                ba big
                ba little
            "#]],
        );
        check(
            r#"#[cfg(target_endian = b$0"#,
            expect![[r#"
                ba big
                ba little
            "#]],
        );
    }
}

mod derive {
    use super::*;

    #[test]
    fn no_completion_for_incorrect_derive() {
        check(
            r#"
//- minicore: derive, copy, clone, ord, eq, default, fmt
#[derive{$0)] struct Test;
"#,
            expect![[]],
        )
    }

    #[test]
    fn empty_derive() {
        check(
            r#"
//- minicore: derive, copy, clone, ord, eq, default, fmt
#[derive($0)] struct Test;
"#,
            expect![[r#"
                de Clone              macro Clone
                de Clone, Copy
                de Default          macro Default
                de PartialEq      macro PartialEq
                de PartialEq, Eq
                de PartialEq, Eq, PartialOrd, Ord
                de PartialEq, PartialOrd
                md core
                kw crate::
                kw self::
            "#]],
        );
    }

    #[test]
    fn derive_with_input_before() {
        check(
            r#"
//- minicore: derive, copy, clone, ord, eq, default, fmt
#[derive(serde::Serialize, PartialEq, $0)] struct Test;
"#,
            expect![[r#"
                de Clone     macro Clone
                de Clone, Copy
                de Default macro Default
                de Eq
                de Eq, PartialOrd, Ord
                de PartialOrd
                md core
                kw crate::
                kw self::
            "#]],
        )
    }

    #[test]
    fn derive_with_input_after() {
        check(
            r#"
//- minicore: derive, copy, clone, ord, eq, default, fmt
#[derive($0 serde::Serialize, PartialEq)] struct Test;
"#,
            expect![[r#"
                de Clone     macro Clone
                de Clone, Copy
                de Default macro Default
                de Eq
                de Eq, PartialOrd, Ord
                de PartialOrd
                md core
                kw crate::
                kw self::
            "#]],
        );
    }

    #[test]
    fn derive_with_existing_derives() {
        check(
            r#"
//- minicore: derive, copy, clone, ord, eq, default, fmt
#[derive(PartialEq, Eq, Or$0)] struct Test;
"#,
            expect![[r#"
                de Clone     macro Clone
                de Clone, Copy
                de Default macro Default
                de PartialOrd
                de PartialOrd, Ord
                md core
                kw crate::
                kw self::
            "#]],
        );
    }

    #[test]
    fn derive_flyimport() {
        check(
            r#"
//- proc_macros: derive_identity
//- minicore: derive
#[derive(der$0)] struct Test;
"#,
            expect![[r#"
                de DeriveIdentity (use proc_macros::DeriveIdentity) proc_macro DeriveIdentity
                md core
                md proc_macros
                kw crate::
                kw self::
            "#]],
        );
        check(
            r#"
//- proc_macros: derive_identity
//- minicore: derive
use proc_macros::DeriveIdentity;
#[derive(der$0)] struct Test;
"#,
            expect![[r#"
                de DeriveIdentity proc_macro DeriveIdentity
                md core
                md proc_macros
                kw crate::
                kw self::
            "#]],
        );
    }

    #[test]
    fn derive_flyimport_edit() {
        check_edit(
            "DeriveIdentity",
            r#"
//- proc_macros: derive_identity
//- minicore: derive
#[derive(der$0)] struct Test;
"#,
            r#"
use proc_macros::DeriveIdentity;

#[derive(DeriveIdentity)] struct Test;
"#,
        );
    }

    #[test]
    fn qualified() {
        check(
            r#"
//- proc_macros: derive_identity
//- minicore: derive, copy, clone
#[derive(proc_macros::$0)] struct Test;
"#,
            expect![[r#"
                de DeriveIdentity proc_macro DeriveIdentity
            "#]],
        );
        check(
            r#"
//- proc_macros: derive_identity
//- minicore: derive, copy, clone
#[derive(proc_macros::C$0)] struct Test;
"#,
            expect![[r#"
                de DeriveIdentity proc_macro DeriveIdentity
            "#]],
        );
    }
}

mod lint {
    use super::*;

    #[test]
    fn lint_empty() {
        check_edit(
            "deprecated",
            r#"#[allow($0)] struct Test;"#,
            r#"#[allow(deprecated)] struct Test;"#,
        )
    }

    #[test]
    fn lint_with_existing() {
        check_edit(
            "deprecated",
            r#"#[allow(keyword_idents, $0)] struct Test;"#,
            r#"#[allow(keyword_idents, deprecated)] struct Test;"#,
        )
    }

    #[test]
    fn lint_qualified() {
        check_edit(
            "deprecated",
            r#"#[allow(keyword_idents, $0)] struct Test;"#,
            r#"#[allow(keyword_idents, deprecated)] struct Test;"#,
        )
    }

    #[test]
    fn lint_feature() {
        check_edit(
            "box_patterns",
            r#"#[feature(box_$0)] struct Test;"#,
            r#"#[feature(box_patterns)] struct Test;"#,
        )
    }

    #[test]
    fn lint_clippy_unqualified() {
        check_edit(
            "clippy::as_conversions",
            r#"#[allow($0)] struct Test;"#,
            r#"#[allow(clippy::as_conversions)] struct Test;"#,
        );
    }

    #[test]
    fn lint_clippy_qualified() {
        check_edit(
            "as_conversions",
            r#"#[allow(clippy::$0)] struct Test;"#,
            r#"#[allow(clippy::as_conversions)] struct Test;"#,
        );
    }

    #[test]
    fn lint_rustdoc_unqualified() {
        check_edit(
            "rustdoc::bare_urls",
            r#"#[allow($0)] struct Test;"#,
            r#"#[allow(rustdoc::bare_urls)] struct Test;"#,
        );
    }

    #[test]
    fn lint_rustdoc_qualified() {
        check_edit(
            "bare_urls",
            r#"#[allow(rustdoc::$0)] struct Test;"#,
            r#"#[allow(rustdoc::bare_urls)] struct Test;"#,
        );
    }

    #[test]
    fn lint_unclosed() {
        check_edit(
            "deprecated",
            r#"#[allow(dep$0 struct Test;"#,
            r#"#[allow(deprecated struct Test;"#,
        );
        check_edit(
            "bare_urls",
            r#"#[allow(rustdoc::$0 struct Test;"#,
            r#"#[allow(rustdoc::bare_urls struct Test;"#,
        );
    }
}

mod repr {
    use super::*;

    #[test]
    fn no_completion_for_incorrect_repr() {
        check(r#"#[repr{$0)] struct Test;"#, expect![[]])
    }

    #[test]
    fn empty() {
        check(
            r#"#[repr($0)] struct Test;"#,
            expect![[r#"
                ba C
                ba align($0)
                ba i16
                ba i28
                ba i32
                ba i64
                ba i8
                ba isize
                ba packed
                ba transparent
                ba u128
                ba u16
                ba u32
                ba u64
                ba u8
                ba usize
            "#]],
        );
    }

    #[test]
    fn transparent() {
        check(r#"#[repr(transparent, $0)] struct Test;"#, expect![[r#""#]]);
    }

    #[test]
    fn align() {
        check(
            r#"#[repr(align(1), $0)] struct Test;"#,
            expect![[r#"
                ba C
                ba i16
                ba i28
                ba i32
                ba i64
                ba i8
                ba isize
                ba transparent
                ba u128
                ba u16
                ba u32
                ba u64
                ba u8
                ba usize
            "#]],
        );
    }

    #[test]
    fn packed() {
        check(
            r#"#[repr(packed, $0)] struct Test;"#,
            expect![[r#"
                ba C
                ba i16
                ba i28
                ba i32
                ba i64
                ba i8
                ba isize
                ba transparent
                ba u128
                ba u16
                ba u32
                ba u64
                ba u8
                ba usize
            "#]],
        );
    }

    #[test]
    fn c() {
        check(
            r#"#[repr(C, $0)] struct Test;"#,
            expect![[r#"
                ba align($0)
                ba i16
                ba i28
                ba i32
                ba i64
                ba i8
                ba isize
                ba packed
                ba u128
                ba u16
                ba u32
                ba u64
                ba u8
                ba usize
            "#]],
        );
    }

    #[test]
    fn prim() {
        check(
            r#"#[repr(usize, $0)] struct Test;"#,
            expect![[r#"
                ba C
                ba align($0)
                ba packed
            "#]],
        );
    }
}

mod macro_use {
    use super::*;

    #[test]
    fn completes_macros() {
        check(
            r#"
//- /dep.rs crate:dep
#[macro_export]
macro_rules! foo {
    () => {};
}

#[macro_export]
macro_rules! bar {
    () => {};
}

//- /main.rs crate:main deps:dep
#[macro_use($0)]
extern crate dep;
"#,
            expect![[r#"
                ma bar
                ma foo
            "#]],
        )
    }

    #[test]
    fn only_completes_exported_macros() {
        check(
            r#"
//- /dep.rs crate:dep
#[macro_export]
macro_rules! foo {
    () => {};
}

macro_rules! bar {
    () => {};
}

//- /main.rs crate:main deps:dep
#[macro_use($0)]
extern crate dep;
"#,
            expect![[r#"
                ma foo
            "#]],
        )
    }

    #[test]
    fn does_not_completes_already_imported_macros() {
        check(
            r#"
//- /dep.rs crate:dep
#[macro_export]
macro_rules! foo {
    () => {};
}

#[macro_export]
macro_rules! bar {
    () => {};
}

//- /main.rs crate:main deps:dep
#[macro_use(foo, $0)]
extern crate dep;
"#,
            expect![[r#"
                ma bar
            "#]],
        )
    }
}
