//! FIXME: write short doc here

use hir::{db::AstDatabase, Adt, HasSource, HirDisplay};
use ra_db::SourceDatabase;
use ra_syntax::{
    algo::find_covering_element,
    ast::{self, DocCommentsOwner},
    match_ast, AstNode,
    SyntaxKind::*,
    SyntaxToken, TokenAtOffset,
};

use crate::{
    db::RootDatabase,
    display::{macro_label, rust_code_markup, rust_code_markup_with_doc, ShortLabel},
    expand::descend_into_macros,
    references::{classify_name, classify_name_ref, NameKind, NameKind::*},
    FilePosition, FileRange, RangeInfo,
};

/// Contains the results when hovering over an item
#[derive(Debug, Clone)]
pub struct HoverResult {
    results: Vec<String>,
    exact: bool,
}

impl Default for HoverResult {
    fn default() -> Self {
        HoverResult::new()
    }
}

impl HoverResult {
    pub fn new() -> HoverResult {
        HoverResult {
            results: Vec::new(),
            // We assume exact by default
            exact: true,
        }
    }

    pub fn extend(&mut self, item: Option<String>) {
        self.results.extend(item);
    }

    pub fn is_exact(&self) -> bool {
        self.exact
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    pub fn len(&self) -> usize {
        self.results.len()
    }

    pub fn first(&self) -> Option<&str> {
        self.results.first().map(String::as_str)
    }

    pub fn results(&self) -> &[String] {
        &self.results
    }

    /// Returns the results converted into markup
    /// for displaying in a UI
    pub fn to_markup(&self) -> String {
        let mut markup = if !self.exact {
            let mut msg = String::from("Failed to exactly resolve the symbol. This is probably because rust_analyzer does not yet support traits.");
            if !self.results.is_empty() {
                msg.push_str("  \nThese items were found instead:");
            }
            msg.push_str("\n\n---\n");
            msg
        } else {
            String::new()
        };

        markup.push_str(&self.results.join("\n\n---\n"));

        markup
    }
}

fn hover_text(docs: Option<String>, desc: Option<String>) -> Option<String> {
    match (desc, docs) {
        (Some(desc), docs) => Some(rust_code_markup_with_doc(desc, docs)),
        (None, Some(docs)) => Some(docs),
        _ => None,
    }
}

fn hover_text_from_name_kind(db: &RootDatabase, name_kind: NameKind) -> Option<String> {
    return match name_kind {
        Macro(it) => {
            let src = it.source(db);
            hover_text(src.value.doc_comment_text(), Some(macro_label(&src.value)))
        }
        Field(it) => {
            let src = it.source(db);
            match src.value {
                hir::FieldSource::Named(it) => hover_text(it.doc_comment_text(), it.short_label()),
                _ => None,
            }
        }
        AssocItem(it) => match it {
            hir::AssocItem::Function(it) => from_def_source(db, it),
            hir::AssocItem::Const(it) => from_def_source(db, it),
            hir::AssocItem::TypeAlias(it) => from_def_source(db, it),
        },
        Def(it) => match it {
            hir::ModuleDef::Module(it) => match it.definition_source(db).value {
                hir::ModuleSource::Module(it) => {
                    hover_text(it.doc_comment_text(), it.short_label())
                }
                _ => None,
            },
            hir::ModuleDef::Function(it) => from_def_source(db, it),
            hir::ModuleDef::Adt(Adt::Struct(it)) => from_def_source(db, it),
            hir::ModuleDef::Adt(Adt::Union(it)) => from_def_source(db, it),
            hir::ModuleDef::Adt(Adt::Enum(it)) => from_def_source(db, it),
            hir::ModuleDef::EnumVariant(it) => from_def_source(db, it),
            hir::ModuleDef::Const(it) => from_def_source(db, it),
            hir::ModuleDef::Static(it) => from_def_source(db, it),
            hir::ModuleDef::Trait(it) => from_def_source(db, it),
            hir::ModuleDef::TypeAlias(it) => from_def_source(db, it),
            hir::ModuleDef::BuiltinType(it) => Some(it.to_string()),
        },
        Local(_) => None,
        TypeParam(_) | SelfType(_) => {
            // FIXME: Hover for generic param
            None
        }
    };

    fn from_def_source<A, D>(db: &RootDatabase, def: D) -> Option<String>
    where
        D: HasSource<Ast = A>,
        A: ast::DocCommentsOwner + ast::NameOwner + ShortLabel,
    {
        let src = def.source(db);
        hover_text(src.value.doc_comment_text(), src.value.short_label())
    }
}

pub(crate) fn hover(db: &RootDatabase, position: FilePosition) -> Option<RangeInfo<HoverResult>> {
    let file = db.parse_or_expand(position.file_id.into())?;
    let token = pick_best(file.token_at_offset(position.offset))?;
    let token = descend_into_macros(db, position.file_id, token);

    let mut res = HoverResult::new();

    if let Some((range, name_kind)) = match_ast! {
        match (token.value.parent()) {
            ast::NameRef(name_ref) => {
                classify_name_ref(db, token.with_value(&name_ref)).map(|d| (name_ref.syntax().text_range(), d.kind))
            },
            ast::Name(name) => {
                classify_name(db, token.with_value(&name)).map(|d| (name.syntax().text_range(), d.kind))
            },
            _ => None,
        }
    } {
        res.extend(hover_text_from_name_kind(db, name_kind));

        if !res.is_empty() {
            return Some(RangeInfo::new(range, res));
        }
    }

    let node = token
        .value
        .ancestors()
        .find(|n| ast::Expr::cast(n.clone()).is_some() || ast::Pat::cast(n.clone()).is_some())?;
    let frange = FileRange { file_id: position.file_id, range: node.text_range() };
    res.extend(type_of(db, frange).map(rust_code_markup));
    if res.is_empty() {
        return None;
    }
    let range = node.text_range();

    Some(RangeInfo::new(range, res))
}

fn pick_best(tokens: TokenAtOffset<SyntaxToken>) -> Option<SyntaxToken> {
    return tokens.max_by_key(priority);
    fn priority(n: &SyntaxToken) -> usize {
        match n.kind() {
            IDENT | INT_NUMBER => 3,
            L_PAREN | R_PAREN => 2,
            kind if kind.is_trivia() => 0,
            _ => 1,
        }
    }
}

pub(crate) fn type_of(db: &RootDatabase, frange: FileRange) -> Option<String> {
    let parse = db.parse(frange.file_id);
    let leaf_node = find_covering_element(parse.tree().syntax(), frange.range);
    // if we picked identifier, expand to pattern/expression
    let node = leaf_node
        .ancestors()
        .take_while(|it| it.text_range() == leaf_node.text_range())
        .find(|it| ast::Expr::cast(it.clone()).is_some() || ast::Pat::cast(it.clone()).is_some())?;
    let analyzer =
        hir::SourceAnalyzer::new(db, hir::InFile::new(frange.file_id.into(), &node), None);
    let ty = if let Some(ty) = ast::Expr::cast(node.clone()).and_then(|e| analyzer.type_of(db, &e))
    {
        ty
    } else if let Some(ty) = ast::Pat::cast(node).and_then(|p| analyzer.type_of_pat(db, &p)) {
        ty
    } else {
        return None;
    };
    Some(ty.display_truncated(db, None).to_string())
}

#[cfg(test)]
mod tests {
    use crate::mock_analysis::{
        analysis_and_position, single_file_with_position, single_file_with_range,
    };
    use ra_syntax::TextRange;

    fn trim_markup(s: &str) -> &str {
        s.trim_start_matches("```rust\n").trim_end_matches("\n```")
    }

    fn trim_markup_opt(s: Option<&str>) -> Option<&str> {
        s.map(trim_markup)
    }

    fn check_hover_result(fixture: &str, expected: &[&str]) {
        let (analysis, position) = analysis_and_position(fixture);
        let hover = analysis.hover(position).unwrap().unwrap();
        let mut results = Vec::from(hover.info.results());
        results.sort();

        for (markup, expected) in
            results.iter().zip(expected.iter().chain(std::iter::repeat(&"<missing>")))
        {
            assert_eq!(trim_markup(&markup), *expected);
        }

        assert_eq!(hover.info.len(), expected.len());
    }

    #[test]
    fn hover_shows_type_of_an_expression() {
        let (analysis, position) = single_file_with_position(
            "
            pub fn foo() -> u32 { 1 }

            fn main() {
                let foo_test = foo()<|>;
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(hover.range, TextRange::from_to(95.into(), 100.into()));
        assert_eq!(trim_markup_opt(hover.info.first()), Some("u32"));
    }

    #[test]
    fn hover_shows_fn_signature() {
        // Single file with result
        check_hover_result(
            r#"
            //- /main.rs
            pub fn foo() -> u32 { 1 }

            fn main() {
                let foo_test = fo<|>o();
            }
        "#,
            &["pub fn foo() -> u32"],
        );

        // Multiple candidates but results are ambiguous.
        check_hover_result(
            r#"
            //- /a.rs
            pub fn foo() -> u32 { 1 }

            //- /b.rs
            pub fn foo() -> &str { "" }

            //- /c.rs
            pub fn foo(a: u32, b: u32) {}

            //- /main.rs
            mod a;
            mod b;
            mod c;

            fn main() {
                let foo_test = fo<|>o();
            }
        "#,
            &["{unknown}"],
        );
    }

    #[test]
    fn hover_shows_fn_signature_with_type_params() {
        check_hover_result(
            r#"
            //- /main.rs
            pub fn foo<'a, T: AsRef<str>>(b: &'a T) -> &'a str { }

            fn main() {
                let foo_test = fo<|>o();
            }
        "#,
            &["pub fn foo<'a, T: AsRef<str>>(b: &'a T) -> &'a str"],
        );
    }

    #[test]
    fn hover_shows_fn_signature_on_fn_name() {
        check_hover_result(
            r#"
            //- /main.rs
            pub fn foo<|>(a: u32, b: u32) -> u32 {}

            fn main() {
            }
        "#,
            &["pub fn foo(a: u32, b: u32) -> u32"],
        );
    }

    #[test]
    fn hover_shows_struct_field_info() {
        // Hovering over the field when instantiating
        check_hover_result(
            r#"
            //- /main.rs
            struct Foo {
                field_a: u32,
            }

            fn main() {
                let foo = Foo {
                    field_a<|>: 0,
                };
            }
        "#,
            &["field_a: u32"],
        );

        // Hovering over the field in the definition
        check_hover_result(
            r#"
            //- /main.rs
            struct Foo {
                field_a<|>: u32,
            }

            fn main() {
                let foo = Foo {
                    field_a: 0,
                };
            }
        "#,
            &["field_a: u32"],
        );
    }

    #[test]
    fn hover_const_static() {
        check_hover_result(
            r#"
            //- /main.rs
            const foo<|>: u32 = 0;
        "#,
            &["const foo: u32"],
        );

        check_hover_result(
            r#"
            //- /main.rs
            static foo<|>: u32 = 0;
        "#,
            &["static foo: u32"],
        );
    }

    #[test]
    fn hover_omits_default_generic_types() {
        check_hover_result(
            r#"
//- /main.rs
struct Test<K, T = u8> {
    k: K,
    t: T,
}

fn main() {
    let zz<|> = Test { t: 23, k: 33 };
}"#,
            &["Test<i32>"],
        );
    }

    #[test]
    fn hover_some() {
        let (analysis, position) = single_file_with_position(
            "
            enum Option<T> { Some(T) }
            use Option::Some;

            fn main() {
                So<|>me(12);
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()), Some("Some"));

        let (analysis, position) = single_file_with_position(
            "
            enum Option<T> { Some(T) }
            use Option::Some;

            fn main() {
                let b<|>ar = Some(12);
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()), Some("Option<i32>"));
    }

    #[test]
    fn hover_enum_variant() {
        check_hover_result(
            r#"
            //- /main.rs
            enum Option<T> {
                /// The None variant
                Non<|>e
            }
        "#,
            &["
None
```

The None variant
            "
            .trim()],
        );

        check_hover_result(
            r#"
            //- /main.rs
            enum Option<T> {
                /// The Some variant
                Some(T)
            }
            fn main() {
                let s = Option::Som<|>e(12);
            }
        "#,
            &["
Some
```

The Some variant
            "
            .trim()],
        );
    }

    #[test]
    fn hover_for_local_variable() {
        let (analysis, position) = single_file_with_position("fn func(foo: i32) { fo<|>o; }");
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()), Some("i32"));
    }

    #[test]
    fn hover_for_local_variable_pat() {
        let (analysis, position) = single_file_with_position("fn func(fo<|>o: i32) {}");
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()), Some("i32"));
    }

    #[test]
    fn hover_local_var_edge() {
        let (analysis, position) = single_file_with_position(
            "
fn func(foo: i32) { if true { <|>foo; }; }
",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()), Some("i32"));
    }

    #[test]
    fn hover_for_param_edge() {
        let (analysis, position) = single_file_with_position("fn func(<|>foo: i32) {}");
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()), Some("i32"));
    }

    #[test]
    fn test_type_of_for_function() {
        let (analysis, range) = single_file_with_range(
            "
            pub fn foo() -> u32 { 1 };

            fn main() {
                let foo_test = <|>foo()<|>;
            }
            ",
        );

        let type_name = analysis.type_of(range).unwrap().unwrap();
        assert_eq!("u32", &type_name);
    }

    #[test]
    fn test_type_of_for_expr() {
        let (analysis, range) = single_file_with_range(
            "
            fn main() {
                let foo: usize = 1;
                let bar = <|>1 + foo<|>;
            }
            ",
        );

        let type_name = analysis.type_of(range).unwrap().unwrap();
        assert_eq!("usize", &type_name);
    }

    #[test]
    fn test_hover_infer_associated_method_result() {
        let (analysis, position) = single_file_with_position(
            "
            struct Thing { x: u32 }

            impl Thing {
                fn new() -> Thing {
                    Thing { x: 0 }
                }
            }

            fn main() {
                let foo_<|>test = Thing::new();
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()), Some("Thing"));
    }

    #[test]
    fn test_hover_infer_associated_method_exact() {
        let (analysis, position) = single_file_with_position(
            "
            struct Thing { x: u32 }

            impl Thing {
                fn new() -> Thing {
                    Thing { x: 0 }
                }
            }

            fn main() {
                let foo_test = Thing::new<|>();
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()), Some("fn new() -> Thing"));
        assert_eq!(hover.info.is_exact(), true);
    }

    #[test]
    fn test_hover_infer_associated_const_in_pattern() {
        let (analysis, position) = single_file_with_position(
            "
            struct X;
            impl X {
                const C: u32 = 1;
            }

            fn main() {
                match 1 {
                    X::C<|> => {},
                    2 => {},
                    _ => {}
                };
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()), Some("const C: u32"));
        assert_eq!(hover.info.is_exact(), true);
    }

    #[test]
    fn test_hover_self() {
        let (analysis, position) = single_file_with_position(
            "
            struct Thing { x: u32 }
            impl Thing {
                fn new() -> Self {
                    Self<|> { x: 0 }
                }
            }
        ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()), Some("Thing"));
        assert_eq!(hover.info.is_exact(), true);

        /* FIXME: revive these tests
                let (analysis, position) = single_file_with_position(
                    "
                    struct Thing { x: u32 }
                    impl Thing {
                        fn new() -> Self<|> {
                            Self { x: 0 }
                        }
                    }
                    ",
                );

                let hover = analysis.hover(position).unwrap().unwrap();
                assert_eq!(trim_markup_opt(hover.info.first()), Some("Thing"));
                assert_eq!(hover.info.is_exact(), true);

                let (analysis, position) = single_file_with_position(
                    "
                    enum Thing { A }
                    impl Thing {
                        pub fn new() -> Self<|> {
                            Thing::A
                        }
                    }
                    ",
                );
                let hover = analysis.hover(position).unwrap().unwrap();
                assert_eq!(trim_markup_opt(hover.info.first()), Some("enum Thing"));
                assert_eq!(hover.info.is_exact(), true);

                let (analysis, position) = single_file_with_position(
                    "
                    enum Thing { A }
                    impl Thing {
                        pub fn thing(a: Self<|>) {
                        }
                    }
                    ",
                );
                let hover = analysis.hover(position).unwrap().unwrap();
                assert_eq!(trim_markup_opt(hover.info.first()), Some("enum Thing"));
                assert_eq!(hover.info.is_exact(), true);
        */
    }

    #[test]
    fn test_hover_shadowing_pat() {
        let (analysis, position) = single_file_with_position(
            "
            fn x() {}

            fn y() {
                let x = 0i32;
                x<|>;
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()), Some("i32"));
        assert_eq!(hover.info.is_exact(), true);
    }

    #[test]
    fn test_hover_macro_invocation() {
        let (analysis, position) = single_file_with_position(
            "
            macro_rules! foo {
                () => {}
            }

            fn f() {
                fo<|>o!();
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()), Some("macro_rules! foo"));
        assert_eq!(hover.info.is_exact(), true);
    }

    #[test]
    fn test_hover_tuple_field() {
        let (analysis, position) = single_file_with_position(
            "
            struct TS(String, i32<|>);
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(trim_markup_opt(hover.info.first()), Some("i32"));
        assert_eq!(hover.info.is_exact(), true);
    }

    #[test]
    fn test_hover_through_macro() {
        check_hover_result(
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
            ",
            &["fn foo()"],
        );
    }
}
