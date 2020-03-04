//! FIXME: write short doc here

use hir::Semantics;
use ra_ide_db::{
    defs::{classify_name, classify_name_ref},
    symbol_index, RootDatabase,
};
use ra_syntax::{
    ast::{self},
    match_ast, AstNode,
    SyntaxKind::*,
    SyntaxToken, TokenAtOffset,
};

use crate::{
    display::{ToNav, TryToNav},
    FilePosition, NavigationTarget, RangeInfo,
};

pub(crate) fn goto_definition(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id).syntax().clone();
    let original_token = pick_best(file.token_at_offset(position.offset))?;
    let token = sema.descend_into_macros(original_token.clone());

    let nav_targets = match_ast! {
        match (token.parent()) {
            ast::NameRef(name_ref) => {
                reference_definition(&sema, &name_ref).to_vec()
            },
            ast::Name(name) => {
                let def = classify_name(&sema, &name)?.definition();
                let nav = def.try_to_nav(sema.db)?;
                vec![nav]
            },
            _ => return None,
        }
    };

    Some(RangeInfo::new(original_token.text_range(), nav_targets))
}

fn pick_best(tokens: TokenAtOffset<SyntaxToken>) -> Option<SyntaxToken> {
    return tokens.max_by_key(priority);
    fn priority(n: &SyntaxToken) -> usize {
        match n.kind() {
            IDENT | INT_NUMBER => 2,
            kind if kind.is_trivia() => 0,
            _ => 1,
        }
    }
}

#[derive(Debug)]
pub(crate) enum ReferenceResult {
    Exact(NavigationTarget),
    Approximate(Vec<NavigationTarget>),
}

impl ReferenceResult {
    fn to_vec(self) -> Vec<NavigationTarget> {
        use self::ReferenceResult::*;
        match self {
            Exact(target) => vec![target],
            Approximate(vec) => vec,
        }
    }
}

pub(crate) fn reference_definition(
    sema: &Semantics<RootDatabase>,
    name_ref: &ast::NameRef,
) -> ReferenceResult {
    use self::ReferenceResult::*;

    let name_kind = classify_name_ref(sema, name_ref);
    if let Some(def) = name_kind {
        let def = def.definition();

        return match def.try_to_nav(sema.db) {
            Some(nav) => ReferenceResult::Exact(nav),
            None => ReferenceResult::Approximate(Vec::new()),
        };
    }

    // Fallback index based approach:
    let navs = symbol_index::index_resolve(sema.db, name_ref)
        .into_iter()
        .map(|s| s.to_nav(sema.db))
        .collect();
    Approximate(navs)
}

#[cfg(test)]
mod tests {
    use test_utils::{assert_eq_text, covers};

    use crate::mock_analysis::analysis_and_position;

    fn check_goto(fixture: &str, expected: &str, expected_range: &str) {
        let (analysis, pos) = analysis_and_position(fixture);

        let mut navs = analysis.goto_definition(pos).unwrap().unwrap().info;
        assert_eq!(navs.len(), 1);

        let nav = navs.pop().unwrap();
        let file_text = analysis.file_text(nav.file_id()).unwrap();

        let mut actual = file_text[nav.full_range()].to_string();
        if let Some(focus) = nav.focus_range() {
            actual += "|";
            actual += &file_text[focus];
        }

        if !expected_range.contains("...") {
            test_utils::assert_eq_text!(&actual, expected_range);
        } else {
            let mut parts = expected_range.split("...");
            let prefix = parts.next().unwrap();
            let suffix = parts.next().unwrap();
            assert!(
                actual.starts_with(prefix) && actual.ends_with(suffix),
                "\nExpected: {}\n Actual: {}\n",
                expected_range,
                actual
            );
        }

        nav.assert_match(expected);
    }

    #[test]
    fn goto_def_in_items() {
        check_goto(
            "
            //- /lib.rs
            struct Foo;
            enum E { X(Foo<|>) }
            ",
            "Foo STRUCT_DEF FileId(1) [0; 11) [7; 10)",
            "struct Foo;|Foo",
        );
    }

    #[test]
    fn goto_def_at_start_of_item() {
        check_goto(
            "
            //- /lib.rs
            struct Foo;
            enum E { X(<|>Foo) }
            ",
            "Foo STRUCT_DEF FileId(1) [0; 11) [7; 10)",
            "struct Foo;|Foo",
        );
    }

    #[test]
    fn goto_definition_resolves_correct_name() {
        check_goto(
            "
            //- /lib.rs
            use a::Foo;
            mod a;
            mod b;
            enum E { X(Foo<|>) }

            //- /a.rs
            struct Foo;

            //- /b.rs
            struct Foo;
            ",
            "Foo STRUCT_DEF FileId(2) [0; 11) [7; 10)",
            "struct Foo;|Foo",
        );
    }

    #[test]
    fn goto_def_for_module_declaration() {
        check_goto(
            "
            //- /lib.rs
            mod <|>foo;

            //- /foo.rs
            // empty
            ",
            "foo SOURCE_FILE FileId(2) [0; 10)",
            "// empty\n\n",
        );

        check_goto(
            "
            //- /lib.rs
            mod <|>foo;

            //- /foo/mod.rs
            // empty
            ",
            "foo SOURCE_FILE FileId(2) [0; 10)",
            "// empty\n\n",
        );
    }

    #[test]
    fn goto_def_for_macros() {
        covers!(ra_ide_db::goto_def_for_macros);
        check_goto(
            "
            //- /lib.rs
            macro_rules! foo { () => { () } }

            fn bar() {
                <|>foo!();
            }
            ",
            "foo MACRO_CALL FileId(1) [0; 33) [13; 16)",
            "macro_rules! foo { () => { () } }|foo",
        );
    }

    #[test]
    fn goto_def_for_macros_from_other_crates() {
        covers!(ra_ide_db::goto_def_for_macros);
        check_goto(
            "
            //- /lib.rs
            use foo::foo;
            fn bar() {
                <|>foo!();
            }

            //- /foo/lib.rs
            #[macro_export]
            macro_rules! foo { () => { () } }
            ",
            "foo MACRO_CALL FileId(2) [0; 49) [29; 32)",
            "#[macro_export]\nmacro_rules! foo { () => { () } }|foo",
        );
    }

    #[test]
    fn goto_def_for_macros_in_use_tree() {
        check_goto(
            "
            //- /lib.rs
            use foo::foo<|>;

            //- /foo/lib.rs
            #[macro_export]
            macro_rules! foo { () => { () } }
            ",
            "foo MACRO_CALL FileId(2) [0; 49) [29; 32)",
            "#[macro_export]\nmacro_rules! foo { () => { () } }|foo",
        );
    }

    #[test]
    fn goto_def_for_macro_defined_fn_with_arg() {
        check_goto(
            "
            //- /lib.rs
            macro_rules! define_fn {
                ($name:ident) => (fn $name() {})
            }

            define_fn!(foo);

            fn bar() {
               <|>foo();
            }
            ",
            "foo FN_DEF FileId(1) [64; 80) [75; 78)",
            "define_fn!(foo);|foo",
        );
    }

    #[test]
    fn goto_def_for_macro_defined_fn_no_arg() {
        check_goto(
            "
            //- /lib.rs
            macro_rules! define_fn {
                () => (fn foo() {})
            }

            define_fn!();

            fn bar() {
               <|>foo();
            }
            ",
            "foo FN_DEF FileId(1) [51; 64) [51; 64)",
            "define_fn!();|define_fn!();",
        );
    }

    #[test]
    fn goto_definition_works_for_macro_inside_pattern() {
        check_goto(
            "
            //- /lib.rs
            macro_rules! foo {() => {0}}

            fn bar() {
                match (0,1) {
                    (<|>foo!(), _) => {}
                }
            }
            ",
            "foo MACRO_CALL FileId(1) [0; 28) [13; 16)",
            "macro_rules! foo {() => {0}}|foo",
        );
    }

    #[test]
    fn goto_definition_works_for_macro_inside_match_arm_lhs() {
        check_goto(
            "
            //- /lib.rs
            macro_rules! foo {() => {0}}

            fn bar() {
                match 0 {
                    <|>foo!() => {}
                }
            }
            ",
            "foo MACRO_CALL FileId(1) [0; 28) [13; 16)",
            "macro_rules! foo {() => {0}}|foo",
        );
    }

    #[test]
    fn goto_def_for_methods() {
        covers!(ra_ide_db::goto_def_for_methods);
        check_goto(
            "
            //- /lib.rs
            struct Foo;
            impl Foo {
                fn frobnicate(&self) { }
            }

            fn bar(foo: &Foo) {
                foo.frobnicate<|>();
            }
            ",
            "frobnicate FN_DEF FileId(1) [27; 51) [30; 40)",
            "fn frobnicate(&self) { }|frobnicate",
        );
    }

    #[test]
    fn goto_def_for_fields() {
        covers!(ra_ide_db::goto_def_for_fields);
        check_goto(
            "
            //- /lib.rs
            struct Foo {
                spam: u32,
            }

            fn bar(foo: &Foo) {
                foo.spam<|>;
            }
            ",
            "spam RECORD_FIELD_DEF FileId(1) [17; 26) [17; 21)",
            "spam: u32|spam",
        );
    }

    #[test]
    fn goto_def_for_record_fields() {
        covers!(ra_ide_db::goto_def_for_record_fields);
        check_goto(
            "
            //- /lib.rs
            struct Foo {
                spam: u32,
            }

            fn bar() -> Foo {
                Foo {
                    spam<|>: 0,
                }
            }
            ",
            "spam RECORD_FIELD_DEF FileId(1) [17; 26) [17; 21)",
            "spam: u32|spam",
        );
    }

    #[test]
    fn goto_for_tuple_fields() {
        check_goto(
            "
            //- /lib.rs
            struct Foo(u32);

            fn bar() {
                let foo = Foo(0);
                foo.<|>0;
            }
            ",
            "TUPLE_FIELD_DEF FileId(1) [11; 14)",
            "u32",
        );
    }

    #[test]
    fn goto_def_for_ufcs_inherent_methods() {
        check_goto(
            "
            //- /lib.rs
            struct Foo;
            impl Foo {
                fn frobnicate() { }
            }

            fn bar(foo: &Foo) {
                Foo::frobnicate<|>();
            }
            ",
            "frobnicate FN_DEF FileId(1) [27; 46) [30; 40)",
            "fn frobnicate() { }|frobnicate",
        );
    }

    #[test]
    fn goto_def_for_ufcs_trait_methods_through_traits() {
        check_goto(
            "
            //- /lib.rs
            trait Foo {
                fn frobnicate();
            }

            fn bar() {
                Foo::frobnicate<|>();
            }
            ",
            "frobnicate FN_DEF FileId(1) [16; 32) [19; 29)",
            "fn frobnicate();|frobnicate",
        );
    }

    #[test]
    fn goto_def_for_ufcs_trait_methods_through_self() {
        check_goto(
            "
            //- /lib.rs
            struct Foo;
            trait Trait {
                fn frobnicate();
            }
            impl Trait for Foo {}

            fn bar() {
                Foo::frobnicate<|>();
            }
            ",
            "frobnicate FN_DEF FileId(1) [30; 46) [33; 43)",
            "fn frobnicate();|frobnicate",
        );
    }

    #[test]
    fn goto_definition_on_self() {
        check_goto(
            "
            //- /lib.rs
            struct Foo;
            impl Foo {
                pub fn new() -> Self {
                    Self<|> {}
                }
            }
            ",
            "impl IMPL_DEF FileId(1) [12; 73)",
            "impl Foo {...}",
        );

        check_goto(
            "
            //- /lib.rs
            struct Foo;
            impl Foo {
                pub fn new() -> Self<|> {
                    Self {}
                }
            }
            ",
            "impl IMPL_DEF FileId(1) [12; 73)",
            "impl Foo {...}",
        );

        check_goto(
            "
            //- /lib.rs
            enum Foo { A }
            impl Foo {
                pub fn new() -> Self<|> {
                    Foo::A
                }
            }
            ",
            "impl IMPL_DEF FileId(1) [15; 75)",
            "impl Foo {...}",
        );

        check_goto(
            "
            //- /lib.rs
            enum Foo { A }
            impl Foo {
                pub fn thing(a: &Self<|>) {
                }
            }
            ",
            "impl IMPL_DEF FileId(1) [15; 62)",
            "impl Foo {...}",
        );
    }

    #[test]
    fn goto_definition_on_self_in_trait_impl() {
        check_goto(
            "
            //- /lib.rs
            struct Foo;
            trait Make {
                fn new() -> Self;
            }
            impl Make for Foo {
                fn new() -> Self {
                    Self<|> {}
                }
            }
            ",
            "impl IMPL_DEF FileId(1) [49; 115)",
            "impl Make for Foo {...}",
        );

        check_goto(
            "
            //- /lib.rs
            struct Foo;
            trait Make {
                fn new() -> Self;
            }
            impl Make for Foo {
                fn new() -> Self<|> {
                    Self {}
                }
            }
            ",
            "impl IMPL_DEF FileId(1) [49; 115)",
            "impl Make for Foo {...}",
        );
    }

    #[test]
    fn goto_def_when_used_on_definition_name_itself() {
        check_goto(
            "
            //- /lib.rs
            struct Foo<|> { value: u32 }
            ",
            "Foo STRUCT_DEF FileId(1) [0; 25) [7; 10)",
            "struct Foo { value: u32 }|Foo",
        );

        check_goto(
            r#"
            //- /lib.rs
            struct Foo {
                field<|>: string,
            }
            "#,
            "field RECORD_FIELD_DEF FileId(1) [17; 30) [17; 22)",
            "field: string|field",
        );

        check_goto(
            "
            //- /lib.rs
            fn foo_test<|>() { }
            ",
            "foo_test FN_DEF FileId(1) [0; 17) [3; 11)",
            "fn foo_test() { }|foo_test",
        );

        check_goto(
            "
            //- /lib.rs
            enum Foo<|> {
                Variant,
            }
            ",
            "Foo ENUM_DEF FileId(1) [0; 25) [5; 8)",
            "enum Foo {...}|Foo",
        );

        check_goto(
            "
            //- /lib.rs
            enum Foo {
                Variant1,
                Variant2<|>,
                Variant3,
            }
            ",
            "Variant2 ENUM_VARIANT FileId(1) [29; 37) [29; 37)",
            "Variant2|Variant2",
        );

        check_goto(
            r#"
            //- /lib.rs
            static INNER<|>: &str = "";
            "#,
            "INNER STATIC_DEF FileId(1) [0; 24) [7; 12)",
            "static INNER: &str = \"\";|INNER",
        );

        check_goto(
            r#"
            //- /lib.rs
            const INNER<|>: &str = "";
            "#,
            "INNER CONST_DEF FileId(1) [0; 23) [6; 11)",
            "const INNER: &str = \"\";|INNER",
        );

        check_goto(
            r#"
            //- /lib.rs
            type Thing<|> = Option<()>;
            "#,
            "Thing TYPE_ALIAS_DEF FileId(1) [0; 24) [5; 10)",
            "type Thing = Option<()>;|Thing",
        );

        check_goto(
            r#"
            //- /lib.rs
            trait Foo<|> { }
            "#,
            "Foo TRAIT_DEF FileId(1) [0; 13) [6; 9)",
            "trait Foo { }|Foo",
        );

        check_goto(
            r#"
            //- /lib.rs
            mod bar<|> { }
            "#,
            "bar MODULE FileId(1) [0; 11) [4; 7)",
            "mod bar { }|bar",
        );
    }

    #[test]
    fn goto_from_macro() {
        check_goto(
            "
            //- /lib.rs
            macro_rules! id {
                ($($tt:tt)*) => { $($tt)* }
            }
            fn foo() {}
            id! {
                fn bar() {
                    fo<|>o();
                }
            }
            mod confuse_index { fn foo(); }
            ",
            "foo FN_DEF FileId(1) [52; 63) [55; 58)",
            "fn foo() {}|foo",
        );
    }

    #[test]
    fn goto_through_format() {
        check_goto(
            "
            //- /lib.rs
            #[macro_export]
            macro_rules! format {
                ($($arg:tt)*) => ($crate::fmt::format($crate::__export::format_args!($($arg)*)))
            }
            #[rustc_builtin_macro]
            #[macro_export]
            macro_rules! format_args {
                ($fmt:expr) => ({ /* compiler built-in */ });
                ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
            }
            pub mod __export {
                pub use crate::format_args;
                fn foo() {} // for index confusion
            }
            fn foo() -> i8 {}
            fn test() {
                format!(\"{}\", fo<|>o())
            }
            ",
            "foo FN_DEF FileId(1) [398; 415) [401; 404)",
            "fn foo() -> i8 {}|foo",
        );
    }

    #[test]
    fn goto_for_type_param() {
        check_goto(
            "
            //- /lib.rs
            struct Foo<T> {
                t: <|>T,
            }
            ",
            "T TYPE_PARAM FileId(1) [11; 12)",
            "T",
        );
    }

    #[test]
    fn goto_within_macro() {
        check_goto(
            "
            //- /lib.rs
            macro_rules! id {
                ($($tt:tt)*) => ($($tt)*)
            }

            fn foo() {
                let x = 1;
                id!({
                    let y = <|>x;
                    let z = y;
                });
            }
            ",
            "x BIND_PAT FileId(1) [69; 70)",
            "x",
        );

        check_goto(
            "
            //- /lib.rs
            macro_rules! id {
                ($($tt:tt)*) => ($($tt)*)
            }

            fn foo() {
                let x = 1;
                id!({
                    let y = x;
                    let z = <|>y;
                });
            }
            ",
            "y BIND_PAT FileId(1) [98; 99)",
            "y",
        );
    }

    #[test]
    fn goto_def_in_local_fn() {
        check_goto(
            "
            //- /lib.rs
            fn main() {
                fn foo() {
                    let x = 92;
                    <|>x;
                }
            }
            ",
            "x BIND_PAT FileId(1) [39; 40)",
            "x",
        );
    }

    #[test]
    fn goto_def_for_field_init_shorthand() {
        covers!(ra_ide_db::goto_def_for_field_init_shorthand);
        check_goto(
            "
            //- /lib.rs
            struct Foo { x: i32 }
            fn main() {
                let x = 92;
                Foo { x<|> };
            }
            ",
            "x BIND_PAT FileId(1) [42; 43)",
            "x",
        )
    }
}
