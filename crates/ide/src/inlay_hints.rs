use std::fmt;

use either::Either;
use hir::{known, HasVisibility, HirDisplay, Semantics};
use ide_db::{base_db::FileRange, famous_defs::FamousDefs, RootDatabase};
use itertools::Itertools;
use syntax::{
    ast::{self, AstNode},
    match_ast, NodeOrToken, SyntaxNode, TextRange, TextSize,
};

use crate::FileId;

mod closing_brace;
mod implicit_static;
mod fn_lifetime_fn;
mod closure_ret;
mod adjustment;
mod chaining;
mod param_name;
mod binding_mode;
mod bind_pat;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InlayHintsConfig {
    pub render_colons: bool,
    pub type_hints: bool,
    pub parameter_hints: bool,
    pub chaining_hints: bool,
    pub adjustment_hints: AdjustmentHints,
    pub closure_return_type_hints: ClosureReturnTypeHints,
    pub binding_mode_hints: bool,
    pub lifetime_elision_hints: LifetimeElisionHints,
    pub param_names_for_lifetime_elision_hints: bool,
    pub hide_named_constructor_hints: bool,
    pub hide_closure_initialization_hints: bool,
    pub max_length: Option<usize>,
    pub closing_brace_hints_min_lines: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClosureReturnTypeHints {
    Always,
    WithBlock,
    Never,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LifetimeElisionHints {
    Always,
    SkipTrivial,
    Never,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdjustmentHints {
    Always,
    ReborrowOnly,
    Never,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InlayKind {
    BindingModeHint,
    ChainingHint,
    ClosingBraceHint,
    ClosureReturnTypeHint,
    GenericParamListHint,
    AdjustmentHint,
    LifetimeHint,
    ParameterHint,
    TypeHint,
    OpeningParenthesis,
    ClosingParenthesis,
}

#[derive(Debug)]
pub struct InlayHint {
    pub range: TextRange,
    pub kind: InlayKind,
    pub label: InlayHintLabel,
    pub tooltip: Option<InlayTooltip>,
}

#[derive(Debug)]
pub enum InlayTooltip {
    String(String),
    HoverRanged(FileId, TextRange),
    HoverOffset(FileId, TextSize),
}

pub struct InlayHintLabel {
    pub parts: Vec<InlayHintLabelPart>,
}

impl InlayHintLabel {
    pub fn as_simple_str(&self) -> Option<&str> {
        match &*self.parts {
            [part] => part.as_simple_str(),
            _ => None,
        }
    }

    pub fn prepend_str(&mut self, s: &str) {
        match &mut *self.parts {
            [part, ..] if part.as_simple_str().is_some() => part.text = format!("{s}{}", part.text),
            _ => self.parts.insert(0, InlayHintLabelPart { text: s.into(), linked_location: None }),
        }
    }

    pub fn append_str(&mut self, s: &str) {
        match &mut *self.parts {
            [.., part] if part.as_simple_str().is_some() => part.text.push_str(s),
            _ => self.parts.push(InlayHintLabelPart { text: s.into(), linked_location: None }),
        }
    }
}

impl From<String> for InlayHintLabel {
    fn from(s: String) -> Self {
        Self { parts: vec![InlayHintLabelPart { text: s, linked_location: None }] }
    }
}

impl From<&str> for InlayHintLabel {
    fn from(s: &str) -> Self {
        Self { parts: vec![InlayHintLabelPart { text: s.into(), linked_location: None }] }
    }
}

impl fmt::Display for InlayHintLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.parts.iter().map(|part| &part.text).format(""))
    }
}

impl fmt::Debug for InlayHintLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(&self.parts).finish()
    }
}

pub struct InlayHintLabelPart {
    pub text: String,
    /// Source location represented by this label part. The client will use this to fetch the part's
    /// hover tooltip, and Ctrl+Clicking the label part will navigate to the definition the location
    /// refers to (not necessarily the location itself).
    /// When setting this, no tooltip must be set on the containing hint, or VS Code will display
    /// them both.
    pub linked_location: Option<FileRange>,
}

impl InlayHintLabelPart {
    pub fn as_simple_str(&self) -> Option<&str> {
        match self {
            Self { text, linked_location: None } => Some(text),
            _ => None,
        }
    }
}

impl fmt::Debug for InlayHintLabelPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_simple_str() {
            Some(string) => string.fmt(f),
            None => f
                .debug_struct("InlayHintLabelPart")
                .field("text", &self.text)
                .field("linked_location", &self.linked_location)
                .finish(),
        }
    }
}

// Feature: Inlay Hints
//
// rust-analyzer shows additional information inline with the source code.
// Editors usually render this using read-only virtual text snippets interspersed with code.
//
// rust-analyzer by default shows hints for
//
// * types of local variables
// * names of function arguments
// * types of chained expressions
//
// Optionally, one can enable additional hints for
//
// * return types of closure expressions
// * elided lifetimes
// * compiler inserted reborrows
//
// image::https://user-images.githubusercontent.com/48062697/113020660-b5f98b80-917a-11eb-8d70-3be3fd558cdd.png[]
pub(crate) fn inlay_hints(
    db: &RootDatabase,
    file_id: FileId,
    range_limit: Option<TextRange>,
    config: &InlayHintsConfig,
) -> Vec<InlayHint> {
    let _p = profile::span("inlay_hints");
    let sema = Semantics::new(db);
    let file = sema.parse(file_id);
    let file = file.syntax();

    let mut acc = Vec::new();

    if let Some(scope) = sema.scope(&file) {
        let famous_defs = FamousDefs(&sema, scope.krate());

        let hints = |node| hints(&mut acc, &famous_defs, config, file_id, node);
        match range_limit {
            Some(range) => match file.covering_element(range) {
                NodeOrToken::Token(_) => return acc,
                NodeOrToken::Node(n) => n
                    .descendants()
                    .filter(|descendant| range.intersect(descendant.text_range()).is_some())
                    .for_each(hints),
            },
            None => file.descendants().for_each(hints),
        };
    }

    acc
}

fn hints(
    hints: &mut Vec<InlayHint>,
    famous_defs @ FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    file_id: FileId,
    node: SyntaxNode,
) {
    closing_brace::hints(hints, sema, config, file_id, node.clone());
    match_ast! {
        match node {
            ast::Expr(expr) => {
                chaining::hints(hints, sema, &famous_defs, config, file_id, &expr);
                adjustment::hints(hints, sema, config, &expr);
                match expr {
                    ast::Expr::CallExpr(it) => param_name::hints(hints, sema, config, ast::Expr::from(it)),
                    ast::Expr::MethodCallExpr(it) => {
                        param_name::hints(hints, sema, config, ast::Expr::from(it))
                    }
                    ast::Expr::ClosureExpr(it) => closure_ret::hints(hints, sema, &famous_defs, config, file_id, it),
                    // We could show reborrows for all expressions, but usually that is just noise to the user
                    // and the main point here is to show why "moving" a mutable reference doesn't necessarily move it
                    // ast::Expr::PathExpr(_) => reborrow_hints(hints, sema, config, &expr),
                    _ => None,
                }
            },
            ast::Pat(it) => {
                binding_mode::hints(hints, sema, config, &it);
                if let ast::Pat::IdentPat(it) = it {
                    bind_pat::hints(hints, sema, config, file_id, &it);
                }
                Some(())
            },
            ast::Item(it) => match it {
                // FIXME: record impl lifetimes so they aren't being reused in assoc item lifetime inlay hints
                ast::Item::Impl(_) => None,
                ast::Item::Fn(it) => fn_lifetime_fn::hints(hints, config, it),
                // static type elisions
                ast::Item::Static(it) => implicit_static::hints(hints, config, Either::Left(it)),
                ast::Item::Const(it) => implicit_static::hints(hints, config, Either::Right(it)),
                _ => None,
            },
            // FIXME: fn-ptr type, dyn fn type, and trait object type elisions
            ast::Type(_) => None,
            _ => None,
        }
    };
}

/// Checks if the type is an Iterator from std::iter and replaces its hint with an `impl Iterator<Item = Ty>`.
fn hint_iterator(
    sema: &Semantics<'_, RootDatabase>,
    famous_defs: &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    ty: &hir::Type,
) -> Option<String> {
    let db = sema.db;
    let strukt = ty.strip_references().as_adt()?;
    let krate = strukt.module(db).krate();
    if krate != famous_defs.core()? {
        return None;
    }
    let iter_trait = famous_defs.core_iter_Iterator()?;
    let iter_mod = famous_defs.core_iter()?;

    // Assert that this struct comes from `core::iter`.
    if !(strukt.visibility(db) == hir::Visibility::Public
        && strukt.module(db).path_to_root(db).contains(&iter_mod))
    {
        return None;
    }

    if ty.impls_trait(db, iter_trait, &[]) {
        let assoc_type_item = iter_trait.items(db).into_iter().find_map(|item| match item {
            hir::AssocItem::TypeAlias(alias) if alias.name(db) == known::Item => Some(alias),
            _ => None,
        })?;
        if let Some(ty) = ty.normalize_trait_assoc_type(db, &[], assoc_type_item) {
            const LABEL_START: &str = "impl Iterator<Item = ";
            const LABEL_END: &str = ">";

            let ty_display = hint_iterator(sema, famous_defs, config, &ty)
                .map(|assoc_type_impl| assoc_type_impl.to_string())
                .unwrap_or_else(|| {
                    ty.display_truncated(
                        db,
                        config
                            .max_length
                            .map(|len| len.saturating_sub(LABEL_START.len() + LABEL_END.len())),
                    )
                    .to_string()
                });
            return Some(format!("{}{}{}", LABEL_START, ty_display, LABEL_END));
        }
    }

    None
}

fn closure_has_block_body(closure: &ast::ClosureExpr) -> bool {
    matches!(closure.body(), Some(ast::Expr::BlockExpr(_)))
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use itertools::Itertools;
    use syntax::{TextRange, TextSize};
    use test_utils::extract_annotations;

    use crate::inlay_hints::AdjustmentHints;
    use crate::{fixture, inlay_hints::InlayHintsConfig, LifetimeElisionHints};

    use super::ClosureReturnTypeHints;

    const DISABLED_CONFIG: InlayHintsConfig = InlayHintsConfig {
        render_colons: false,
        type_hints: false,
        parameter_hints: false,
        chaining_hints: false,
        lifetime_elision_hints: LifetimeElisionHints::Never,
        closure_return_type_hints: ClosureReturnTypeHints::Never,
        adjustment_hints: AdjustmentHints::Never,
        binding_mode_hints: false,
        hide_named_constructor_hints: false,
        hide_closure_initialization_hints: false,
        param_names_for_lifetime_elision_hints: false,
        max_length: None,
        closing_brace_hints_min_lines: None,
    };
    const TEST_CONFIG: InlayHintsConfig = InlayHintsConfig {
        type_hints: true,
        parameter_hints: true,
        chaining_hints: true,
        closure_return_type_hints: ClosureReturnTypeHints::WithBlock,
        binding_mode_hints: true,
        lifetime_elision_hints: LifetimeElisionHints::Always,
        ..DISABLED_CONFIG
    };

    #[track_caller]
    fn check(ra_fixture: &str) {
        check_with_config(TEST_CONFIG, ra_fixture);
    }

    #[track_caller]
    fn check_params(ra_fixture: &str) {
        check_with_config(
            InlayHintsConfig { parameter_hints: true, ..DISABLED_CONFIG },
            ra_fixture,
        );
    }

    #[track_caller]
    fn check_types(ra_fixture: &str) {
        check_with_config(InlayHintsConfig { type_hints: true, ..DISABLED_CONFIG }, ra_fixture);
    }

    #[track_caller]
    fn check_chains(ra_fixture: &str) {
        check_with_config(InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG }, ra_fixture);
    }

    #[track_caller]
    fn check_with_config(config: InlayHintsConfig, ra_fixture: &str) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let mut expected = extract_annotations(&*analysis.file_text(file_id).unwrap());
        let inlay_hints = analysis.inlay_hints(&config, file_id, None).unwrap();
        let actual = inlay_hints
            .into_iter()
            .map(|it| (it.range, it.label.to_string()))
            .sorted_by_key(|(range, _)| range.start())
            .collect::<Vec<_>>();
        expected.sort_by_key(|(range, _)| range.start());

        assert_eq!(expected, actual, "\nExpected:\n{:#?}\n\nActual:\n{:#?}", expected, actual);
    }

    #[track_caller]
    fn check_expect(config: InlayHintsConfig, ra_fixture: &str, expect: Expect) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let inlay_hints = analysis.inlay_hints(&config, file_id, None).unwrap();
        expect.assert_debug_eq(&inlay_hints)
    }

    #[test]
    fn hints_disabled() {
        check_with_config(
            InlayHintsConfig { render_colons: true, ..DISABLED_CONFIG },
            r#"
fn foo(a: i32, b: i32) -> i32 { a + b }
fn main() {
    let _x = foo(4, 4);
}"#,
        );
    }

    // Parameter hint tests

    #[test]
    fn param_hints_only() {
        check_params(
            r#"
fn foo(a: i32, b: i32) -> i32 { a + b }
fn main() {
    let _x = foo(
        4,
      //^ a
        4,
      //^ b
    );
}"#,
        );
    }

    #[test]
    fn param_hints_on_closure() {
        check_params(
            r#"
fn main() {
    let clo = |a: u8, b: u8| a + b;
    clo(
        1,
      //^ a
        2,
      //^ b
    );
}
            "#,
        );
    }

    #[test]
    fn param_name_similar_to_fn_name_still_hints() {
        check_params(
            r#"
fn max(x: i32, y: i32) -> i32 { x + y }
fn main() {
    let _x = max(
        4,
      //^ x
        4,
      //^ y
    );
}"#,
        );
    }

    #[test]
    fn param_name_similar_to_fn_name() {
        check_params(
            r#"
fn param_with_underscore(with_underscore: i32) -> i32 { with_underscore }
fn main() {
    let _x = param_with_underscore(
        4,
    );
}"#,
        );
        check_params(
            r#"
fn param_with_underscore(underscore: i32) -> i32 { underscore }
fn main() {
    let _x = param_with_underscore(
        4,
    );
}"#,
        );
    }

    #[test]
    fn param_name_same_as_fn_name() {
        check_params(
            r#"
fn foo(foo: i32) -> i32 { foo }
fn main() {
    let _x = foo(
        4,
    );
}"#,
        );
    }

    #[test]
    fn never_hide_param_when_multiple_params() {
        check_params(
            r#"
fn foo(foo: i32, bar: i32) -> i32 { bar + baz }
fn main() {
    let _x = foo(
        4,
      //^ foo
        8,
      //^ bar
    );
}"#,
        );
    }

    #[test]
    fn param_hints_look_through_as_ref_and_clone() {
        check_params(
            r#"
fn foo(bar: i32, baz: f32) {}

fn main() {
    let bar = 3;
    let baz = &"baz";
    let fez = 1.0;
    foo(bar.clone(), bar.clone());
                   //^^^^^^^^^^^ baz
    foo(bar.as_ref(), bar.as_ref());
                    //^^^^^^^^^^^^ baz
}
"#,
        );
    }

    #[test]
    fn self_param_hints() {
        check_params(
            r#"
struct Foo;

impl Foo {
    fn foo(self: Self) {}
    fn bar(self: &Self) {}
}

fn main() {
    Foo::foo(Foo);
           //^^^ self
    Foo::bar(&Foo);
           //^^^^ self
}
"#,
        )
    }

    #[test]
    fn param_name_hints_show_for_literals() {
        check_params(
            r#"pub fn test(a: i32, b: i32) -> [i32; 2] { [a, b] }
fn main() {
    test(
        0xa_b,
      //^^^^^ a
        0xa_b,
      //^^^^^ b
    );
}"#,
        )
    }

    #[test]
    fn function_call_parameter_hint() {
        check_params(
            r#"
//- minicore: option
struct FileId {}
struct SmolStr {}

struct TextRange {}
struct SyntaxKind {}
struct NavigationTarget {}

struct Test {}

impl Test {
    fn method(&self, mut param: i32) -> i32 { param * 2 }

    fn from_syntax(
        file_id: FileId,
        name: SmolStr,
        focus_range: Option<TextRange>,
        full_range: TextRange,
        kind: SyntaxKind,
        docs: Option<String>,
    ) -> NavigationTarget {
        NavigationTarget {}
    }
}

fn test_func(mut foo: i32, bar: i32, msg: &str, _: i32, last: i32) -> i32 {
    foo + bar
}

fn main() {
    let not_literal = 1;
    let _: i32 = test_func(1,    2,      "hello", 3,  not_literal);
                         //^ foo ^ bar   ^^^^^^^ msg  ^^^^^^^^^^^ last
    let t: Test = Test {};
    t.method(123);
           //^^^ param
    Test::method(&t,      3456);
               //^^ self  ^^^^ param
    Test::from_syntax(
        FileId {},
        "impl".into(),
      //^^^^^^^^^^^^^ name
        None,
      //^^^^ focus_range
        TextRange {},
      //^^^^^^^^^^^^ full_range
        SyntaxKind {},
      //^^^^^^^^^^^^^ kind
        None,
      //^^^^ docs
    );
}"#,
        );
    }

    #[test]
    fn parameter_hint_heuristics() {
        check_params(
            r#"
fn check(ra_fixture_thing: &str) {}

fn map(f: i32) {}
fn filter(predicate: i32) {}

fn strip_suffix(suffix: &str) {}
fn stripsuffix(suffix: &str) {}
fn same(same: u32) {}
fn same2(_same2: u32) {}

fn enum_matches_param_name(completion_kind: CompletionKind) {}

fn foo(param: u32) {}
fn bar(param_eter: u32) {}

enum CompletionKind {
    Keyword,
}

fn non_ident_pat((a, b): (u32, u32)) {}

fn main() {
    const PARAM: u32 = 0;
    foo(PARAM);
    foo(!PARAM);
     // ^^^^^^ param
    check("");

    map(0);
    filter(0);

    strip_suffix("");
    stripsuffix("");
              //^^ suffix
    same(0);
    same2(0);

    enum_matches_param_name(CompletionKind::Keyword);

    let param = 0;
    foo(param);
    foo(param as _);
    let param_end = 0;
    foo(param_end);
    let start_param = 0;
    foo(start_param);
    let param2 = 0;
    foo(param2);
      //^^^^^^ param

    macro_rules! param {
        () => {};
    };
    foo(param!());

    let param_eter = 0;
    bar(param_eter);
    let param_eter_end = 0;
    bar(param_eter_end);
    let start_param_eter = 0;
    bar(start_param_eter);
    let param_eter2 = 0;
    bar(param_eter2);
      //^^^^^^^^^^^ param_eter

    non_ident_pat((0, 0));
}"#,
        );
    }

    // Type-Hint tests

    #[test]
    fn type_hints_only() {
        check_types(
            r#"
fn foo(a: i32, b: i32) -> i32 { a + b }
fn main() {
    let _x = foo(4, 4);
      //^^ i32
}"#,
        );
    }

    #[test]
    fn type_hints_bindings_after_at() {
        check_types(
            r#"
//- minicore: option
fn main() {
    let ref foo @ bar @ ref mut baz = 0;
          //^^^ &i32
                //^^^ i32
                              //^^^ &mut i32
    let [x @ ..] = [0];
       //^ [i32; 1]
    if let x @ Some(_) = Some(0) {}
         //^ Option<i32>
    let foo @ (bar, baz) = (3, 3);
      //^^^ (i32, i32)
             //^^^ i32
                  //^^^ i32
}"#,
        );
    }

    #[test]
    fn default_generic_types_should_not_be_displayed() {
        check(
            r#"
struct Test<K, T = u8> { k: K, t: T }

fn main() {
    let zz = Test { t: 23u8, k: 33 };
      //^^ Test<i32>
    let zz_ref = &zz;
      //^^^^^^ &Test<i32>
    let test = || zz;
      //^^^^ || -> Test<i32>
}"#,
        );
    }

    #[test]
    fn shorten_iterators_in_associated_params() {
        check_types(
            r#"
//- minicore: iterators
use core::iter;

pub struct SomeIter<T> {}

impl<T> SomeIter<T> {
    pub fn new() -> Self { SomeIter {} }
    pub fn push(&mut self, t: T) {}
}

impl<T> Iterator for SomeIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() {
    let mut some_iter = SomeIter::new();
          //^^^^^^^^^ SomeIter<Take<Repeat<i32>>>
      some_iter.push(iter::repeat(2).take(2));
    let iter_of_iters = some_iter.take(2);
      //^^^^^^^^^^^^^ impl Iterator<Item = impl Iterator<Item = i32>>
}
"#,
        );
    }

    #[test]
    fn iterator_hint_regression_issue_12674() {
        // Ensure we don't crash while solving the projection type of iterators.
        check_expect(
            InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: iterators
struct S<T>(T);
impl<T> S<T> {
    fn iter(&self) -> Iter<'_, T> { loop {} }
}
struct Iter<'a, T: 'a>(&'a T);
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> { loop {} }
}
struct Container<'a> {
    elements: S<&'a str>,
}
struct SliceIter<'a, T>(&'a T);
impl<'a, T> Iterator for SliceIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> { loop {} }
}

fn main(a: SliceIter<'_, Container>) {
    a
    .filter_map(|c| Some(c.elements.iter().filter_map(|v| Some(v))))
    .map(|e| e);
}
            "#,
            expect![[r#"
                [
                    InlayHint {
                        range: 484..554,
                        kind: ChainingHint,
                        label: [
                            "impl Iterator<Item = impl Iterator<Item = &&str>>",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                484..554,
                            ),
                        ),
                    },
                    InlayHint {
                        range: 484..485,
                        kind: ChainingHint,
                        label: [
                            "SliceIter<Container>",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                484..485,
                            ),
                        ),
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn infer_call_method_return_associated_types_with_generic() {
        check_types(
            r#"
            pub trait Default {
                fn default() -> Self;
            }
            pub trait Foo {
                type Bar: Default;
            }

            pub fn quux<T: Foo>() -> T::Bar {
                let y = Default::default();
                  //^ <T as Foo>::Bar

                y
            }
            "#,
        );
    }

    #[test]
    fn fn_hints() {
        check_types(
            r#"
//- minicore: fn, sized
fn foo() -> impl Fn() { loop {} }
fn foo1() -> impl Fn(f64) { loop {} }
fn foo2() -> impl Fn(f64, f64) { loop {} }
fn foo3() -> impl Fn(f64, f64) -> u32 { loop {} }
fn foo4() -> &'static dyn Fn(f64, f64) -> u32 { loop {} }
fn foo5() -> &'static dyn Fn(&'static dyn Fn(f64, f64) -> u32, f64) -> u32 { loop {} }
fn foo6() -> impl Fn(f64, f64) -> u32 + Sized { loop {} }
fn foo7() -> *const (impl Fn(f64, f64) -> u32 + Sized) { loop {} }

fn main() {
    let foo = foo();
     // ^^^ impl Fn()
    let foo = foo1();
     // ^^^ impl Fn(f64)
    let foo = foo2();
     // ^^^ impl Fn(f64, f64)
    let foo = foo3();
     // ^^^ impl Fn(f64, f64) -> u32
    let foo = foo4();
     // ^^^ &dyn Fn(f64, f64) -> u32
    let foo = foo5();
     // ^^^ &dyn Fn(&dyn Fn(f64, f64) -> u32, f64) -> u32
    let foo = foo6();
     // ^^^ impl Fn(f64, f64) -> u32
    let foo = foo7();
     // ^^^ *const impl Fn(f64, f64) -> u32
}
"#,
        )
    }

    #[test]
    fn check_hint_range_limit() {
        let fixture = r#"
        //- minicore: fn, sized
        fn foo() -> impl Fn() { loop {} }
        fn foo1() -> impl Fn(f64) { loop {} }
        fn foo2() -> impl Fn(f64, f64) { loop {} }
        fn foo3() -> impl Fn(f64, f64) -> u32 { loop {} }
        fn foo4() -> &'static dyn Fn(f64, f64) -> u32 { loop {} }
        fn foo5() -> &'static dyn Fn(&'static dyn Fn(f64, f64) -> u32, f64) -> u32 { loop {} }
        fn foo6() -> impl Fn(f64, f64) -> u32 + Sized { loop {} }
        fn foo7() -> *const (impl Fn(f64, f64) -> u32 + Sized) { loop {} }

        fn main() {
            let foo = foo();
            let foo = foo1();
            let foo = foo2();
             // ^^^ impl Fn(f64, f64)
            let foo = foo3();
             // ^^^ impl Fn(f64, f64) -> u32
            let foo = foo4();
            let foo = foo5();
            let foo = foo6();
            let foo = foo7();
        }
        "#;
        let (analysis, file_id) = fixture::file(fixture);
        let expected = extract_annotations(&*analysis.file_text(file_id).unwrap());
        let inlay_hints = analysis
            .inlay_hints(
                &InlayHintsConfig { type_hints: true, ..DISABLED_CONFIG },
                file_id,
                Some(TextRange::new(TextSize::from(500), TextSize::from(600))),
            )
            .unwrap();
        let actual =
            inlay_hints.into_iter().map(|it| (it.range, it.label.to_string())).collect::<Vec<_>>();
        assert_eq!(expected, actual, "\nExpected:\n{:#?}\n\nActual:\n{:#?}", expected, actual);
    }

    #[test]
    fn fn_hints_ptr_rpit_fn_parentheses() {
        check_types(
            r#"
//- minicore: fn, sized
trait Trait {}

fn foo1() -> *const impl Fn() { loop {} }
fn foo2() -> *const (impl Fn() + Sized) { loop {} }
fn foo3() -> *const (impl Fn() + ?Sized) { loop {} }
fn foo4() -> *const (impl Sized + Fn()) { loop {} }
fn foo5() -> *const (impl ?Sized + Fn()) { loop {} }
fn foo6() -> *const (impl Fn() + Trait) { loop {} }
fn foo7() -> *const (impl Fn() + Sized + Trait) { loop {} }
fn foo8() -> *const (impl Fn() + ?Sized + Trait) { loop {} }
fn foo9() -> *const (impl Fn() -> u8 + ?Sized) { loop {} }
fn foo10() -> *const (impl Fn() + Sized + ?Sized) { loop {} }

fn main() {
    let foo = foo1();
    //  ^^^ *const impl Fn()
    let foo = foo2();
    //  ^^^ *const impl Fn()
    let foo = foo3();
    //  ^^^ *const (impl Fn() + ?Sized)
    let foo = foo4();
    //  ^^^ *const impl Fn()
    let foo = foo5();
    //  ^^^ *const (impl Fn() + ?Sized)
    let foo = foo6();
    //  ^^^ *const (impl Fn() + Trait)
    let foo = foo7();
    //  ^^^ *const (impl Fn() + Trait)
    let foo = foo8();
    //  ^^^ *const (impl Fn() + Trait + ?Sized)
    let foo = foo9();
    //  ^^^ *const (impl Fn() -> u8 + ?Sized)
    let foo = foo10();
    //  ^^^ *const impl Fn()
}
"#,
        )
    }

    #[test]
    fn unit_structs_have_no_type_hints() {
        check_types(
            r#"
//- minicore: result
struct SyntheticSyntax;

fn main() {
    match Ok(()) {
        Ok(_) => (),
        Err(SyntheticSyntax) => (),
    }
}"#,
        );
    }

    #[test]
    fn let_statement() {
        check_types(
            r#"
#[derive(PartialEq)]
enum Option<T> { None, Some(T) }

#[derive(PartialEq)]
struct Test { a: Option<u32>, b: u8 }

fn main() {
    struct InnerStruct {}

    let test = 54;
      //^^^^ i32
    let test: i32 = 33;
    let mut test = 33;
          //^^^^ i32
    let _ = 22;
    let test = "test";
      //^^^^ &str
    let test = InnerStruct {};
      //^^^^ InnerStruct

    let test = unresolved();

    let test = (42, 'a');
      //^^^^ (i32, char)
    let (a,    (b,     (c,)) = (2, (3, (9.2,));
       //^ i32  ^ i32   ^ f64
    let &x = &92;
       //^ i32
}"#,
        );
    }

    #[test]
    fn if_expr() {
        check_types(
            r#"
//- minicore: option
struct Test { a: Option<u32>, b: u8 }

fn main() {
    let test = Some(Test { a: Some(3), b: 1 });
      //^^^^ Option<Test>
    if let None = &test {};
    if let test = &test {};
         //^^^^ &Option<Test>
    if let Some(test) = &test {};
              //^^^^ &Test
    if let Some(Test { a,             b }) = &test {};
                     //^ &Option<u32> ^ &u8
    if let Some(Test { a: x,             b: y }) = &test {};
                        //^ &Option<u32>    ^ &u8
    if let Some(Test { a: Some(x),  b: y }) = &test {};
                             //^ &u32  ^ &u8
    if let Some(Test { a: None,  b: y }) = &test {};
                                  //^ &u8
    if let Some(Test { b: y, .. }) = &test {};
                        //^ &u8
    if test == None {}
}"#,
        );
    }

    #[test]
    fn while_expr() {
        check_types(
            r#"
//- minicore: option
struct Test { a: Option<u32>, b: u8 }

fn main() {
    let test = Some(Test { a: Some(3), b: 1 });
      //^^^^ Option<Test>
    while let Some(Test { a: Some(x),  b: y }) = &test {};
                                //^ &u32  ^ &u8
}"#,
        );
    }

    #[test]
    fn match_arm_list() {
        check_types(
            r#"
//- minicore: option
struct Test { a: Option<u32>, b: u8 }

fn main() {
    match Some(Test { a: Some(3), b: 1 }) {
        None => (),
        test => (),
      //^^^^ Option<Test>
        Some(Test { a: Some(x), b: y }) => (),
                          //^ u32  ^ u8
        _ => {}
    }
}"#,
        );
    }

    #[test]
    fn complete_for_hint() {
        check_types(
            r#"
//- minicore: iterator
pub struct Vec<T> {}

impl<T> Vec<T> {
    pub fn new() -> Self { Vec {} }
    pub fn push(&mut self, t: T) {}
}

impl<T> IntoIterator for Vec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
}

struct IntoIter<T> {}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
}

fn main() {
    let mut data = Vec::new();
          //^^^^ Vec<&str>
    data.push("foo");
    for i in data {
      //^ &str
      let z = i;
        //^ &str
    }
}
"#,
        );
    }

    #[test]
    fn multi_dyn_trait_bounds() {
        check_types(
            r#"
pub struct Vec<T> {}

impl<T> Vec<T> {
    pub fn new() -> Self { Vec {} }
}

pub struct Box<T> {}

trait Display {}
auto trait Sync {}

fn main() {
    // The block expression wrapping disables the constructor hint hiding logic
    let _v = { Vec::<Box<&(dyn Display + Sync)>>::new() };
      //^^ Vec<Box<&(dyn Display + Sync)>>
    let _v = { Vec::<Box<*const (dyn Display + Sync)>>::new() };
      //^^ Vec<Box<*const (dyn Display + Sync)>>
    let _v = { Vec::<Box<dyn Display + Sync>>::new() };
      //^^ Vec<Box<dyn Display + Sync>>
}
"#,
        );
    }

    #[test]
    fn shorten_iterator_hints() {
        check_types(
            r#"
//- minicore: iterators
use core::iter;

struct MyIter;

impl Iterator for MyIter {
    type Item = ();
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() {
    let _x = MyIter;
      //^^ MyIter
    let _x = iter::repeat(0);
      //^^ impl Iterator<Item = i32>
    fn generic<T: Clone>(t: T) {
        let _x = iter::repeat(t);
          //^^ impl Iterator<Item = T>
        let _chained = iter::repeat(t).take(10);
          //^^^^^^^^ impl Iterator<Item = T>
    }
}
"#,
        );
    }

    #[test]
    fn skip_constructor_and_enum_type_hints() {
        check_with_config(
            InlayHintsConfig {
                type_hints: true,
                hide_named_constructor_hints: true,
                ..DISABLED_CONFIG
            },
            r#"
//- minicore: try, option
use core::ops::ControlFlow;

mod x {
    pub mod y { pub struct Foo; }
    pub struct Foo;
    pub enum AnotherEnum {
        Variant()
    };
}
struct Struct;
struct TupleStruct();

impl Struct {
    fn new() -> Self {
        Struct
    }
    fn try_new() -> ControlFlow<(), Self> {
        ControlFlow::Continue(Struct)
    }
}

struct Generic<T>(T);
impl Generic<i32> {
    fn new() -> Self {
        Generic(0)
    }
}

enum Enum {
    Variant(u32)
}

fn times2(value: i32) -> i32 {
    2 * value
}

fn main() {
    let enumb = Enum::Variant(0);

    let strukt = x::Foo;
    let strukt = x::y::Foo;
    let strukt = Struct;
    let strukt = Struct::new();

    let tuple_struct = TupleStruct();

    let generic0 = Generic::new();
    //  ^^^^^^^^ Generic<i32>
    let generic1 = Generic(0);
    //  ^^^^^^^^ Generic<i32>
    let generic2 = Generic::<i32>::new();
    let generic3 = <Generic<i32>>::new();
    let generic4 = Generic::<i32>(0);


    let option = Some(0);
    //  ^^^^^^ Option<i32>
    let func = times2;
    //  ^^^^ fn times2(i32) -> i32
    let closure = |x: i32| x * 2;
    //  ^^^^^^^ |i32| -> i32
}

fn fallible() -> ControlFlow<()> {
    let strukt = Struct::try_new()?;
}
"#,
        );
    }

    #[test]
    fn shows_constructor_type_hints_when_enabled() {
        check_types(
            r#"
//- minicore: try
use core::ops::ControlFlow;

struct Struct;
struct TupleStruct();

impl Struct {
    fn new() -> Self {
        Struct
    }
    fn try_new() -> ControlFlow<(), Self> {
        ControlFlow::Continue(Struct)
    }
}

struct Generic<T>(T);
impl Generic<i32> {
    fn new() -> Self {
        Generic(0)
    }
}

fn main() {
    let strukt = Struct::new();
     // ^^^^^^ Struct
    let tuple_struct = TupleStruct();
     // ^^^^^^^^^^^^ TupleStruct
    let generic0 = Generic::new();
     // ^^^^^^^^ Generic<i32>
    let generic1 = Generic::<i32>::new();
     // ^^^^^^^^ Generic<i32>
    let generic2 = <Generic<i32>>::new();
     // ^^^^^^^^ Generic<i32>
}

fn fallible() -> ControlFlow<()> {
    let strukt = Struct::try_new()?;
     // ^^^^^^ Struct
}
"#,
        );
    }

    #[test]
    fn closures() {
        check(
            r#"
fn main() {
    let mut start = 0;
          //^^^^^ i32
    (0..2).for_each(|increment      | { start += increment; });
                   //^^^^^^^^^ i32

    let multiply =
      //^^^^^^^^ |i32, i32| -> i32
      | a,     b| a * b
      //^ i32  ^ i32

    ;

    let _: i32 = multiply(1,  2);
                        //^ a ^ b
    let multiply_ref = &multiply;
      //^^^^^^^^^^^^ &|i32, i32| -> i32

    let return_42 = || 42;
      //^^^^^^^^^ || -> i32
      || { 42 };
    //^^ i32
}"#,
        );
    }

    #[test]
    fn return_type_hints_for_closure_without_block() {
        check_with_config(
            InlayHintsConfig {
                closure_return_type_hints: ClosureReturnTypeHints::Always,
                ..DISABLED_CONFIG
            },
            r#"
fn main() {
    let a = || { 0 };
          //^^ i32
    let b = || 0;
          //^^ i32
}"#,
        );
    }

    #[test]
    fn skip_closure_type_hints() {
        check_with_config(
            InlayHintsConfig {
                type_hints: true,
                hide_closure_initialization_hints: true,
                ..DISABLED_CONFIG
            },
            r#"
//- minicore: fn
fn main() {
    let multiple_2 = |x: i32| { x * 2 };

    let multiple_2 = |x: i32| x * 2;
    //  ^^^^^^^^^^ |i32| -> i32

    let (not) = (|x: bool| { !x });
    //   ^^^ |bool| -> bool

    let (is_zero, _b) = (|x: usize| { x == 0 }, false);
    //   ^^^^^^^ |usize| -> bool
    //            ^^ bool

    let plus_one = |x| { x + 1 };
    //              ^ u8
    foo(plus_one);

    let add_mul = bar(|x: u8| { x + 1 });
    //  ^^^^^^^ impl FnOnce(u8) -> u8 + ?Sized

    let closure = if let Some(6) = add_mul(2).checked_sub(1) {
    //  ^^^^^^^ fn(i32) -> i32
        |x: i32| { x * 2 }
    } else {
        |x: i32| { x * 3 }
    };
}

fn foo(f: impl FnOnce(u8) -> u8) {}

fn bar(f: impl FnOnce(u8) -> u8) -> impl FnOnce(u8) -> u8 {
    move |x: u8| f(x) * 2
}
"#,
        );
    }

    #[test]
    fn hint_truncation() {
        check_with_config(
            InlayHintsConfig { max_length: Some(8), ..TEST_CONFIG },
            r#"
struct Smol<T>(T);

struct VeryLongOuterName<T>(T);

fn main() {
    let a = Smol(0u32);
      //^ Smol<u32>
    let b = VeryLongOuterName(0usize);
      //^ VeryLongOuterName<…>
    let c = Smol(Smol(0u32))
      //^ Smol<Smol<…>>
}"#,
        );
    }

    // Chaining hint tests

    #[test]
    fn chaining_hints_ignore_comments() {
        check_expect(
            InlayHintsConfig { type_hints: false, chaining_hints: true, ..DISABLED_CONFIG },
            r#"
struct A(B);
impl A { fn into_b(self) -> B { self.0 } }
struct B(C);
impl B { fn into_c(self) -> C { self.0 } }
struct C;

fn main() {
    let c = A(B(C))
        .into_b() // This is a comment
        // This is another comment
        .into_c();
}
"#,
            expect![[r#"
                [
                    InlayHint {
                        range: 147..172,
                        kind: ChainingHint,
                        label: [
                            "B",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                147..172,
                            ),
                        ),
                    },
                    InlayHint {
                        range: 147..154,
                        kind: ChainingHint,
                        label: [
                            "A",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                147..154,
                            ),
                        ),
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn chaining_hints_without_newlines() {
        check_chains(
            r#"
struct A(B);
impl A { fn into_b(self) -> B { self.0 } }
struct B(C);
impl B { fn into_c(self) -> C { self.0 } }
struct C;

fn main() {
    let c = A(B(C)).into_b().into_c();
}"#,
        );
    }

    #[test]
    fn struct_access_chaining_hints() {
        check_expect(
            InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG },
            r#"
struct A { pub b: B }
struct B { pub c: C }
struct C(pub bool);
struct D;

impl D {
    fn foo(&self) -> i32 { 42 }
}

fn main() {
    let x = A { b: B { c: C(true) } }
        .b
        .c
        .0;
    let x = D
        .foo();
}"#,
            expect![[r#"
                [
                    InlayHint {
                        range: 143..190,
                        kind: ChainingHint,
                        label: [
                            "C",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                143..190,
                            ),
                        ),
                    },
                    InlayHint {
                        range: 143..179,
                        kind: ChainingHint,
                        label: [
                            "B",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                143..179,
                            ),
                        ),
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn generic_chaining_hints() {
        check_expect(
            InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG },
            r#"
struct A<T>(T);
struct B<T>(T);
struct C<T>(T);
struct X<T,R>(T, R);

impl<T> A<T> {
    fn new(t: T) -> Self { A(t) }
    fn into_b(self) -> B<T> { B(self.0) }
}
impl<T> B<T> {
    fn into_c(self) -> C<T> { C(self.0) }
}
fn main() {
    let c = A::new(X(42, true))
        .into_b()
        .into_c();
}
"#,
            expect![[r#"
                [
                    InlayHint {
                        range: 246..283,
                        kind: ChainingHint,
                        label: [
                            "B<X<i32, bool>>",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                246..283,
                            ),
                        ),
                    },
                    InlayHint {
                        range: 246..265,
                        kind: ChainingHint,
                        label: [
                            "A<X<i32, bool>>",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                246..265,
                            ),
                        ),
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn shorten_iterator_chaining_hints() {
        check_expect(
            InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: iterators
use core::iter;

struct MyIter;

impl Iterator for MyIter {
    type Item = ();
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() {
    let _x = MyIter.by_ref()
        .take(5)
        .by_ref()
        .take(5)
        .by_ref();
}
"#,
            expect![[r#"
                [
                    InlayHint {
                        range: 174..241,
                        kind: ChainingHint,
                        label: [
                            "impl Iterator<Item = ()>",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                174..241,
                            ),
                        ),
                    },
                    InlayHint {
                        range: 174..224,
                        kind: ChainingHint,
                        label: [
                            "impl Iterator<Item = ()>",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                174..224,
                            ),
                        ),
                    },
                    InlayHint {
                        range: 174..206,
                        kind: ChainingHint,
                        label: [
                            "impl Iterator<Item = ()>",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                174..206,
                            ),
                        ),
                    },
                    InlayHint {
                        range: 174..189,
                        kind: ChainingHint,
                        label: [
                            "&mut MyIter",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                174..189,
                            ),
                        ),
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn hints_in_attr_call() {
        check_expect(
            TEST_CONFIG,
            r#"
//- proc_macros: identity, input_replace
struct Struct;
impl Struct {
    fn chain(self) -> Self {
        self
    }
}
#[proc_macros::identity]
fn main() {
    let strukt = Struct;
    strukt
        .chain()
        .chain()
        .chain();
    Struct::chain(strukt);
}
"#,
            expect![[r#"
                [
                    InlayHint {
                        range: 124..130,
                        kind: TypeHint,
                        label: [
                            "Struct",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                124..130,
                            ),
                        ),
                    },
                    InlayHint {
                        range: 145..185,
                        kind: ChainingHint,
                        label: [
                            "Struct",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                145..185,
                            ),
                        ),
                    },
                    InlayHint {
                        range: 145..168,
                        kind: ChainingHint,
                        label: [
                            "Struct",
                        ],
                        tooltip: Some(
                            HoverRanged(
                                FileId(
                                    0,
                                ),
                                145..168,
                            ),
                        ),
                    },
                    InlayHint {
                        range: 222..228,
                        kind: ParameterHint,
                        label: [
                            "self",
                        ],
                        tooltip: Some(
                            HoverOffset(
                                FileId(
                                    0,
                                ),
                                42,
                            ),
                        ),
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn hints_lifetimes() {
        check(
            r#"
fn empty() {}

fn no_gpl(a: &()) {}
 //^^^^^^<'0>
          // ^'0
fn empty_gpl<>(a: &()) {}
      //    ^'0   ^'0
fn partial<'b>(a: &(), b: &'b ()) {}
//        ^'0, $  ^'0
fn partial<'a>(a: &'a (), b: &()) {}
//        ^'0, $             ^'0

fn single_ret(a: &()) -> &() {}
// ^^^^^^^^^^<'0>
              // ^'0     ^'0
fn full_mul(a: &(), b: &()) {}
// ^^^^^^^^<'0, '1>
            // ^'0     ^'1

fn foo<'c>(a: &'c ()) -> &() {}
                      // ^'c

fn nested_in(a: &   &X< &()>) {}
// ^^^^^^^^^<'0, '1, '2>
              //^'0 ^'1 ^'2
fn nested_out(a: &()) -> &   &X< &()>{}
// ^^^^^^^^^^<'0>
               //^'0     ^'0 ^'0 ^'0

impl () {
    fn foo(&self) {}
    // ^^^<'0>
        // ^'0
    fn foo(&self) -> &() {}
    // ^^^<'0>
        // ^'0       ^'0
    fn foo(&self, a: &()) -> &() {}
    // ^^^<'0, '1>
        // ^'0       ^'1     ^'0
}
"#,
        );
    }

    #[test]
    fn hints_lifetimes_named() {
        check_with_config(
            InlayHintsConfig { param_names_for_lifetime_elision_hints: true, ..TEST_CONFIG },
            r#"
fn nested_in<'named>(named: &        &X<      &()>) {}
//          ^'named1, 'named2, 'named3, $
                          //^'named1 ^'named2 ^'named3
"#,
        );
    }

    #[test]
    fn hints_lifetimes_trivial_skip() {
        check_with_config(
            InlayHintsConfig {
                lifetime_elision_hints: LifetimeElisionHints::SkipTrivial,
                ..TEST_CONFIG
            },
            r#"
fn no_gpl(a: &()) {}
fn empty_gpl<>(a: &()) {}
fn partial<'b>(a: &(), b: &'b ()) {}
fn partial<'a>(a: &'a (), b: &()) {}

fn single_ret(a: &()) -> &() {}
// ^^^^^^^^^^<'0>
              // ^'0     ^'0
fn full_mul(a: &(), b: &()) {}

fn foo<'c>(a: &'c ()) -> &() {}
                      // ^'c

fn nested_in(a: &   &X< &()>) {}
fn nested_out(a: &()) -> &   &X< &()>{}
// ^^^^^^^^^^<'0>
               //^'0     ^'0 ^'0 ^'0

impl () {
    fn foo(&self) {}
    fn foo(&self) -> &() {}
    // ^^^<'0>
        // ^'0       ^'0
    fn foo(&self, a: &()) -> &() {}
    // ^^^<'0, '1>
        // ^'0       ^'1     ^'0
}
"#,
        );
    }

    #[test]
    fn hints_lifetimes_static() {
        check_with_config(
            InlayHintsConfig {
                lifetime_elision_hints: LifetimeElisionHints::Always,
                ..TEST_CONFIG
            },
            r#"
trait Trait {}
static S: &str = "";
//        ^'static
const C: &str = "";
//       ^'static
const C: &dyn Trait = panic!();
//       ^'static

impl () {
    const C: &str = "";
    const C: &dyn Trait = panic!();
}
"#,
        );
    }

    #[test]
    fn hints_binding_modes() {
        check_with_config(
            InlayHintsConfig { binding_mode_hints: true, ..DISABLED_CONFIG },
            r#"
fn __(
    (x,): (u32,),
    (x,): &(u32,),
  //^^^^&
   //^ ref
    (x,): &mut (u32,)
  //^^^^&mut
   //^ ref mut
) {
    let (x,) = (0,);
    let (x,) = &(0,);
      //^^^^ &
       //^ ref
    let (x,) = &mut (0,);
      //^^^^ &mut
       //^ ref mut
    let &mut (x,) = &mut (0,);
    let (ref mut x,) = &mut (0,);
      //^^^^^^^^^^^^ &mut
    let &mut (ref mut x,) = &mut (0,);
    let (mut x,) = &mut (0,);
      //^^^^^^^^ &mut
    match (0,) {
        (x,) => ()
    }
    match &(0,) {
        (x,) | (x,) => (),
      //^^^^^^^^^^^&
       //^ ref
              //^ ref
      //^^^^^^^^^^^(
      //^^^^^^^^^^^)
        ((x,) | (x,)) => (),
      //^^^^^^^^^^^^^&
        //^ ref
               //^ ref
    }
    match &mut (0,) {
        (x,) => ()
      //^^^^ &mut
       //^ ref mut
    }
}"#,
        );
    }

    #[test]
    fn hints_closing_brace() {
        check_with_config(
            InlayHintsConfig { closing_brace_hints_min_lines: Some(2), ..DISABLED_CONFIG },
            r#"
fn a() {}

fn f() {
} // no hint unless `}` is the last token on the line

fn g() {
  }
//^ fn g

fn h<T>(with: T, arguments: u8, ...) {
  }
//^ fn h

trait Tr {
    fn f();
    fn g() {
    }
  //^ fn g
  }
//^ trait Tr
impl Tr for () {
  }
//^ impl Tr for ()
impl dyn Tr {
  }
//^ impl dyn Tr

static S0: () = 0;
static S1: () = {};
static S2: () = {
 };
//^ static S2
const _: () = {
 };
//^ const _

mod m {
  }
//^ mod m

m! {}
m!();
m!(
 );
//^ m!

m! {
  }
//^ m!

fn f() {
    let v = vec![
    ];
  }
//^ fn f
"#,
        );
    }

    #[test]
    fn adjustment_hints() {
        check_with_config(
            InlayHintsConfig { adjustment_hints: AdjustmentHints::Always, ..DISABLED_CONFIG },
            r#"
//- minicore: coerce_unsized
fn main() {
    let _: u32         = loop {};
                       //^^^^^^^<never-to-any>
    let _: &u32        = &mut 0;
                       //^^^^^^&
                       //^^^^^^*
    let _: &mut u32    = &mut 0;
                       //^^^^^^&mut $
                       //^^^^^^*
    let _: *const u32  = &mut 0;
                       //^^^^^^&raw const $
                       //^^^^^^*
    let _: *mut u32    = &mut 0;
                       //^^^^^^&raw mut $
                       //^^^^^^*
    let _: fn()        = main;
                       //^^^^<fn-item-to-fn-pointer>
    let _: unsafe fn() = main;
                       //^^^^<safe-fn-pointer-to-unsafe-fn-pointer>
                       //^^^^<fn-item-to-fn-pointer>
    let _: unsafe fn() = main as fn();
                       //^^^^^^^^^^^^<safe-fn-pointer-to-unsafe-fn-pointer>
    let _: fn()        = || {};
                       //^^^^^<closure-to-fn-pointer>
    let _: unsafe fn() = || {};
                       //^^^^^<closure-to-unsafe-fn-pointer>
    let _: *const u32  = &mut 0u32 as *mut u32;
                       //^^^^^^^^^^^^^^^^^^^^^<mut-ptr-to-const-ptr>
    let _: &mut [_]    = &mut [0; 0];
                       //^^^^^^^^^^^<unsize>
                       //^^^^^^^^^^^&mut $
                       //^^^^^^^^^^^*

    Struct.consume();
    Struct.by_ref();
  //^^^^^^(
  //^^^^^^&
  //^^^^^^)
    Struct.by_ref_mut();
  //^^^^^^(
  //^^^^^^&mut $
  //^^^^^^)

    (&Struct).consume();
   //^^^^^^^*
    (&Struct).by_ref();

    (&mut Struct).consume();
   //^^^^^^^^^^^*
    (&mut Struct).by_ref();
   //^^^^^^^^^^^&
   //^^^^^^^^^^^*
    (&mut Struct).by_ref_mut();

    // Check that block-like expressions don't duplicate hints
    let _: &mut [u32] = (&mut []);
                       //^^^^^^^<unsize>
                       //^^^^^^^&mut $
                       //^^^^^^^*
    let _: &mut [u32] = { &mut [] };
                        //^^^^^^^<unsize>
                        //^^^^^^^&mut $
                        //^^^^^^^*
    let _: &mut [u32] = unsafe { &mut [] };
                               //^^^^^^^<unsize>
                               //^^^^^^^&mut $
                               //^^^^^^^*
    let _: &mut [u32] = if true {
        &mut []
      //^^^^^^^<unsize>
      //^^^^^^^&mut $
      //^^^^^^^*
    } else {
        loop {}
      //^^^^^^^<never-to-any>
    };
    let _: &mut [u32] = match () { () => &mut [] }
                                       //^^^^^^^<unsize>
                                       //^^^^^^^&mut $
                                       //^^^^^^^*
}

#[derive(Copy, Clone)]
struct Struct;
impl Struct {
    fn consume(self) {}
    fn by_ref(&self) {}
    fn by_ref_mut(&mut self) {}
}
trait Trait {}
impl Trait for Struct {}
"#,
        )
    }
}
