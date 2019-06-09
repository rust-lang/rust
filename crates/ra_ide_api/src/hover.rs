use ra_db::SourceDatabase;
use ra_syntax::{
    AstNode, ast,
    algo::{find_covering_element, find_node_at_offset, ancestors_at_offset},
};
use hir::HirDisplay;

use crate::{
    db::RootDatabase,
    RangeInfo, FilePosition, FileRange,
    display::{rust_code_markup, doc_text_for},
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

pub(crate) fn hover(db: &RootDatabase, position: FilePosition) -> Option<RangeInfo<HoverResult>> {
    let file = db.parse(position.file_id).tree;
    let mut res = HoverResult::new();

    let mut range = None;
    if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(file.syntax(), position.offset) {
        use crate::goto_definition::{ReferenceResult::*, reference_definition};
        let ref_result = reference_definition(db, position.file_id, name_ref);
        match ref_result {
            Exact(nav) => res.extend(doc_text_for(nav)),
            Approximate(navs) => {
                // We are no longer exact
                res.exact = false;

                for nav in navs {
                    res.extend(doc_text_for(nav))
                }
            }
        }
        if !res.is_empty() {
            range = Some(name_ref.syntax().range())
        }
    } else if let Some(name) = find_node_at_offset::<ast::Name>(file.syntax(), position.offset) {
        let navs = crate::goto_definition::name_definition(db, position.file_id, name);

        if let Some(navs) = navs {
            for nav in navs {
                res.extend(doc_text_for(nav))
            }
        }

        if !res.is_empty() && range.is_none() {
            range = Some(name.syntax().range());
        }
    }

    if range.is_none() {
        let node = ancestors_at_offset(file.syntax(), position.offset)
            .find(|n| ast::Expr::cast(*n).is_some() || ast::Pat::cast(*n).is_some())?;
        let frange = FileRange { file_id: position.file_id, range: node.range() };
        res.extend(type_of(db, frange).map(rust_code_markup));
        range = Some(node.range());
    }

    let range = range?;
    if res.is_empty() {
        return None;
    }
    let res = RangeInfo::new(range, res);
    Some(res)
}

pub(crate) fn type_of(db: &RootDatabase, frange: FileRange) -> Option<String> {
    let file = db.parse(frange.file_id).tree;
    let syntax = file.syntax();
    let leaf_node = find_covering_element(syntax, frange.range);
    // if we picked identifier, expand to pattern/expression
    let node = leaf_node
        .ancestors()
        .take_while(|it| it.range() == leaf_node.range())
        .find(|&it| ast::Expr::cast(it).is_some() || ast::Pat::cast(it).is_some())?;
    let analyzer = hir::SourceAnalyzer::new(db, frange.file_id, node, None);
    let ty = if let Some(ty) = ast::Expr::cast(node).and_then(|e| analyzer.type_of(db, e)) {
        ty
    } else if let Some(ty) = ast::Pat::cast(node).and_then(|p| analyzer.type_of_pat(db, p)) {
        ty
    } else {
        return None;
    };
    Some(ty.display(db).to_string())
}

#[cfg(test)]
mod tests {
    use ra_syntax::TextRange;
    use crate::mock_analysis::{single_file_with_position, single_file_with_range, analysis_and_position};

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

        // Multiple results
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
            &["pub fn foo() -> &str", "pub fn foo() -> u32", "pub fn foo(a: u32, b: u32)"],
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
            fn main() {
                const foo<|>: u32 = 0;
            }
        "#,
            &["const foo: u32"],
        );

        check_hover_result(
            r#"
            //- /main.rs
            fn main() {
                static foo<|>: u32 = 0;
            }
        "#,
            &["static foo: u32"],
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
        assert_eq!(trim_markup_opt(hover.info.first()), Some("struct Thing"));
        assert_eq!(hover.info.is_exact(), true);

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
        assert_eq!(trim_markup_opt(hover.info.first()), Some("struct Thing"));
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
    }
}
