use base_db::SourceDatabase;
use hir::{
    Adt, AsAssocItem, AssocItemContainer, Documentation, FieldSource, HasSource, HirDisplay,
    Module, ModuleDef, ModuleSource, Semantics,
};
use ide_db::{
    defs::{classify_name, classify_name_ref, Definition},
    RootDatabase,
};
use itertools::Itertools;
use stdx::format_to;
use syntax::{ast, match_ast, AstNode, SyntaxKind::*, SyntaxToken, TokenAtOffset, T};
use test_utils::mark;

use crate::{
    display::{macro_label, ShortLabel, ToNav, TryToNav},
    link_rewrite::rewrite_links,
    markup::Markup,
    runnables::runnable,
    FileId, FilePosition, NavigationTarget, RangeInfo, Runnable,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HoverConfig {
    pub implementations: bool,
    pub run: bool,
    pub debug: bool,
    pub goto_type_def: bool,
}

impl Default for HoverConfig {
    fn default() -> Self {
        Self { implementations: true, run: true, debug: true, goto_type_def: true }
    }
}

impl HoverConfig {
    pub const NO_ACTIONS: Self =
        Self { implementations: false, run: false, debug: false, goto_type_def: false };

    pub fn any(&self) -> bool {
        self.implementations || self.runnable() || self.goto_type_def
    }

    pub fn none(&self) -> bool {
        !self.any()
    }

    pub fn runnable(&self) -> bool {
        self.run || self.debug
    }
}

#[derive(Debug, Clone)]
pub enum HoverAction {
    Runnable(Runnable),
    Implementaion(FilePosition),
    GoToType(Vec<HoverGotoTypeData>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct HoverGotoTypeData {
    pub mod_path: String,
    pub nav: NavigationTarget,
}

/// Contains the results when hovering over an item
#[derive(Debug, Default)]
pub struct HoverResult {
    pub markup: Markup,
    pub actions: Vec<HoverAction>,
}

// Feature: Hover
//
// Shows additional information, like type of an expression or documentation for definition when "focusing" code.
// Focusing is usually hovering with a mouse, but can also be triggered with a shortcut.
pub(crate) fn hover(db: &RootDatabase, position: FilePosition) -> Option<RangeInfo<HoverResult>> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id).syntax().clone();
    let token = pick_best(file.token_at_offset(position.offset))?;
    let token = sema.descend_into_macros(token);

    let mut res = HoverResult::default();

    let node = token.parent();
    let definition = match_ast! {
        match node {
            ast::NameRef(name_ref) => classify_name_ref(&sema, &name_ref).map(|d| d.definition(sema.db)),
            ast::Name(name) => classify_name(&sema, &name).map(|d| d.definition(sema.db)),
            _ => None,
        }
    };
    if let Some(definition) = definition {
        if let Some(markup) = hover_for_definition(db, definition) {
            let markup = rewrite_links(db, &markup.as_str(), &definition);
            res.markup = Markup::from(markup);
            if let Some(action) = show_implementations_action(db, definition) {
                res.actions.push(action);
            }

            if let Some(action) = runnable_action(&sema, definition, position.file_id) {
                res.actions.push(action);
            }

            if let Some(action) = goto_type_action(db, definition) {
                res.actions.push(action);
            }

            let range = sema.original_range(&node).range;
            return Some(RangeInfo::new(range, res));
        }
    }

    let node = token
        .ancestors()
        .find(|n| ast::Expr::cast(n.clone()).is_some() || ast::Pat::cast(n.clone()).is_some())?;

    let ty = match_ast! {
        match node {
            ast::Expr(it) => sema.type_of_expr(&it)?,
            ast::Pat(it) => sema.type_of_pat(&it)?,
            // If this node is a MACRO_CALL, it means that `descend_into_macros` failed to resolve.
            // (e.g expanding a builtin macro). So we give up here.
            ast::MacroCall(_it) => return None,
            _ => return None,
        }
    };

    res.markup = Markup::fenced_block(&ty.display(db));
    let range = sema.original_range(&node).range;
    Some(RangeInfo::new(range, res))
}

fn show_implementations_action(db: &RootDatabase, def: Definition) -> Option<HoverAction> {
    fn to_action(nav_target: NavigationTarget) -> HoverAction {
        HoverAction::Implementaion(FilePosition {
            file_id: nav_target.file_id,
            offset: nav_target.focus_or_full_range().start(),
        })
    }

    match def {
        Definition::ModuleDef(it) => match it {
            ModuleDef::Adt(Adt::Struct(it)) => Some(to_action(it.to_nav(db))),
            ModuleDef::Adt(Adt::Union(it)) => Some(to_action(it.to_nav(db))),
            ModuleDef::Adt(Adt::Enum(it)) => Some(to_action(it.to_nav(db))),
            ModuleDef::Trait(it) => Some(to_action(it.to_nav(db))),
            _ => None,
        },
        _ => None,
    }
}

fn runnable_action(
    sema: &Semantics<RootDatabase>,
    def: Definition,
    file_id: FileId,
) -> Option<HoverAction> {
    match def {
        Definition::ModuleDef(it) => match it {
            ModuleDef::Module(it) => match it.definition_source(sema.db).value {
                ModuleSource::Module(it) => runnable(&sema, it.syntax().clone(), file_id)
                    .map(|it| HoverAction::Runnable(it)),
                _ => None,
            },
            ModuleDef::Function(it) => {
                let src = it.source(sema.db);
                if src.file_id != file_id.into() {
                    mark::hit!(hover_macro_generated_struct_fn_doc_comment);
                    mark::hit!(hover_macro_generated_struct_fn_doc_attr);

                    return None;
                }

                runnable(&sema, src.value.syntax().clone(), file_id)
                    .map(|it| HoverAction::Runnable(it))
            }
            _ => None,
        },
        _ => None,
    }
}

fn goto_type_action(db: &RootDatabase, def: Definition) -> Option<HoverAction> {
    match def {
        Definition::Local(it) => {
            let mut targets: Vec<ModuleDef> = Vec::new();
            let mut push_new_def = |item: ModuleDef| {
                if !targets.contains(&item) {
                    targets.push(item);
                }
            };

            it.ty(db).walk(db, |t| {
                if let Some(adt) = t.as_adt() {
                    push_new_def(adt.into());
                } else if let Some(trait_) = t.as_dyn_trait() {
                    push_new_def(trait_.into());
                } else if let Some(traits) = t.as_impl_traits(db) {
                    traits.into_iter().for_each(|it| push_new_def(it.into()));
                } else if let Some(trait_) = t.as_associated_type_parent_trait(db) {
                    push_new_def(trait_.into());
                }
            });

            let targets = targets
                .into_iter()
                .filter_map(|it| {
                    Some(HoverGotoTypeData {
                        mod_path: render_path(
                            db,
                            it.module(db)?,
                            it.name(db).map(|name| name.to_string()),
                        ),
                        nav: it.try_to_nav(db)?,
                    })
                })
                .collect();

            Some(HoverAction::GoToType(targets))
        }
        _ => None,
    }
}

fn hover_markup(
    docs: Option<String>,
    desc: Option<String>,
    mod_path: Option<String>,
) -> Option<Markup> {
    match desc {
        Some(desc) => {
            let mut buf = String::new();

            if let Some(mod_path) = mod_path {
                if !mod_path.is_empty() {
                    format_to!(buf, "```rust\n{}\n```\n\n", mod_path);
                }
            }
            format_to!(buf, "```rust\n{}\n```", desc);

            if let Some(doc) = docs {
                format_to!(buf, "\n___\n\n{}", doc);
            }
            Some(buf.into())
        }
        None => docs.map(Markup::from),
    }
}

fn definition_owner_name(db: &RootDatabase, def: &Definition) -> Option<String> {
    match def {
        Definition::Field(f) => Some(f.parent_def(db).name(db)),
        Definition::Local(l) => l.parent(db).name(db),
        Definition::ModuleDef(md) => match md {
            ModuleDef::Function(f) => match f.as_assoc_item(db)?.container(db) {
                AssocItemContainer::Trait(t) => Some(t.name(db)),
                AssocItemContainer::ImplDef(i) => i.target_ty(db).as_adt().map(|adt| adt.name(db)),
            },
            ModuleDef::EnumVariant(e) => Some(e.parent_enum(db).name(db)),
            _ => None,
        },
        Definition::SelfType(i) => i.target_ty(db).as_adt().map(|adt| adt.name(db)),
        _ => None,
    }
    .map(|name| name.to_string())
}

fn render_path(db: &RootDatabase, module: Module, item_name: Option<String>) -> String {
    let crate_name =
        db.crate_graph()[module.krate().into()].display_name.as_ref().map(ToString::to_string);
    let module_path = module
        .path_to_root(db)
        .into_iter()
        .rev()
        .flat_map(|it| it.name(db).map(|name| name.to_string()));
    crate_name.into_iter().chain(module_path).chain(item_name).join("::")
}

fn definition_mod_path(db: &RootDatabase, def: &Definition) -> Option<String> {
    def.module(db).map(|module| render_path(db, module, definition_owner_name(db, def)))
}

fn hover_for_definition(db: &RootDatabase, def: Definition) -> Option<Markup> {
    let mod_path = definition_mod_path(db, &def);
    return match def {
        Definition::Macro(it) => {
            let src = it.source(db);
            let docs = Documentation::from_ast(&src.value).map(Into::into);
            hover_markup(docs, Some(macro_label(&src.value)), mod_path)
        }
        Definition::Field(it) => {
            let src = it.source(db);
            match src.value {
                FieldSource::Named(it) => {
                    let docs = Documentation::from_ast(&it).map(Into::into);
                    hover_markup(docs, it.short_label(), mod_path)
                }
                _ => None,
            }
        }
        Definition::ModuleDef(it) => match it {
            ModuleDef::Module(it) => match it.definition_source(db).value {
                ModuleSource::Module(it) => {
                    let docs = Documentation::from_ast(&it).map(Into::into);
                    hover_markup(docs, it.short_label(), mod_path)
                }
                ModuleSource::SourceFile(it) => {
                    let docs = Documentation::from_ast(&it).map(Into::into);
                    hover_markup(docs, it.short_label(), mod_path)
                }
            },
            ModuleDef::Function(it) => from_def_source(db, it, mod_path),
            ModuleDef::Adt(Adt::Struct(it)) => from_def_source(db, it, mod_path),
            ModuleDef::Adt(Adt::Union(it)) => from_def_source(db, it, mod_path),
            ModuleDef::Adt(Adt::Enum(it)) => from_def_source(db, it, mod_path),
            ModuleDef::EnumVariant(it) => from_def_source(db, it, mod_path),
            ModuleDef::Const(it) => from_def_source(db, it, mod_path),
            ModuleDef::Static(it) => from_def_source(db, it, mod_path),
            ModuleDef::Trait(it) => from_def_source(db, it, mod_path),
            ModuleDef::TypeAlias(it) => from_def_source(db, it, mod_path),
            ModuleDef::BuiltinType(it) => return Some(it.to_string().into()),
        },
        Definition::Local(it) => return Some(Markup::fenced_block(&it.ty(db).display(db))),
        Definition::TypeParam(_) | Definition::SelfType(_) => {
            // FIXME: Hover for generic param
            None
        }
    };

    fn from_def_source<A, D>(db: &RootDatabase, def: D, mod_path: Option<String>) -> Option<Markup>
    where
        D: HasSource<Ast = A>,
        A: ast::DocCommentsOwner + ast::NameOwner + ShortLabel + ast::AttrsOwner,
    {
        let src = def.source(db);
        let docs = Documentation::from_ast(&src.value).map(Into::into);
        hover_markup(docs, src.value.short_label(), mod_path)
    }
}

fn pick_best(tokens: TokenAtOffset<SyntaxToken>) -> Option<SyntaxToken> {
    return tokens.max_by_key(priority);
    fn priority(n: &SyntaxToken) -> usize {
        match n.kind() {
            IDENT | INT_NUMBER => 3,
            T!['('] | T![')'] => 2,
            kind if kind.is_trivia() => 0,
            _ => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use base_db::FileLoader;
    use expect_test::{expect, Expect};

    use crate::mock_analysis::analysis_and_position;

    use super::*;

    fn check_hover_no_result(ra_fixture: &str) {
        let (analysis, position) = analysis_and_position(ra_fixture);
        assert!(analysis.hover(position).unwrap().is_none());
    }

    fn check(ra_fixture: &str, expect: Expect) {
        let (analysis, position) = analysis_and_position(ra_fixture);
        let hover = analysis.hover(position).unwrap().unwrap();

        let content = analysis.db.file_text(position.file_id);
        let hovered_element = &content[hover.range];

        let actual = format!("*{}*\n{}\n", hovered_element, hover.info.markup);
        expect.assert_eq(&actual)
    }

    fn check_actions(ra_fixture: &str, expect: Expect) {
        let (analysis, position) = analysis_and_position(ra_fixture);
        let hover = analysis.hover(position).unwrap().unwrap();
        expect.assert_debug_eq(&hover.info.actions)
    }

    #[test]
    fn hover_shows_type_of_an_expression() {
        check(
            r#"
pub fn foo() -> u32 { 1 }

fn main() {
    let foo_test = foo()<|>;
}
"#,
            expect![[r#"
                *foo()*
                ```rust
                u32
                ```
            "#]],
        );
    }

    #[test]
    fn hover_shows_long_type_of_an_expression() {
        check(
            r#"
struct Scan<A, B, C> { a: A, b: B, c: C }
struct Iter<I> { inner: I }
enum Option<T> { Some(T), None }

struct OtherStruct<T> { i: T }

fn scan<A, B, C>(a: A, b: B, c: C) -> Iter<Scan<OtherStruct<A>, B, C>> {
    Iter { inner: Scan { a, b, c } }
}

fn main() {
    let num: i32 = 55;
    let closure = |memo: &mut u32, value: &u32, _another: &mut u32| -> Option<u32> {
        Option::Some(*memo + value)
    };
    let number = 5u32;
    let mut iter<|> = scan(OtherStruct { i: num }, closure, number);
}
"#,
            expect![[r#"
                *iter*

                ```rust
                Iter<Scan<OtherStruct<OtherStruct<i32>>, |&mut u32, &u32, &mut u32| -> Option<u32>, u32>>
                ```
            "#]],
        );
    }

    #[test]
    fn hover_shows_fn_signature() {
        // Single file with result
        check(
            r#"
pub fn foo() -> u32 { 1 }

fn main() { let foo_test = fo<|>o(); }
"#,
            expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                pub fn foo() -> u32
                ```
            "#]],
        );

        // Multiple candidates but results are ambiguous.
        check(
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

fn main() { let foo_test = fo<|>o(); }
        "#,
            expect![[r#"
                *foo*
                ```rust
                {unknown}
                ```
            "#]],
        );
    }

    #[test]
    fn hover_shows_fn_signature_with_type_params() {
        check(
            r#"
pub fn foo<'a, T: AsRef<str>>(b: &'a T) -> &'a str { }

fn main() { let foo_test = fo<|>o(); }
        "#,
            expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                pub fn foo<'a, T: AsRef<str>>(b: &'a T) -> &'a str
                ```
            "#]],
        );
    }

    #[test]
    fn hover_shows_fn_signature_on_fn_name() {
        check(
            r#"
pub fn foo<|>(a: u32, b: u32) -> u32 {}

fn main() { }
"#,
            expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                pub fn foo(a: u32, b: u32) -> u32
                ```
            "#]],
        );
    }

    #[test]
    fn hover_shows_fn_doc() {
        check(
            r#"
/// # Example
/// ```
/// # use std::path::Path;
/// #
/// foo(Path::new("hello, world!"))
/// ```
pub fn foo<|>(_: &Path) {}

fn main() { }
"#,
            expect![[r##"
                *foo*

                ```rust
                test
                ```

                ```rust
                pub fn foo(_: &Path)
                ```

                ---

                # Example

                ```
                # use std::path::Path;
                #
                foo(Path::new("hello, world!"))
                ```
            "##]],
        );
    }

    #[test]
    fn hover_shows_struct_field_info() {
        // Hovering over the field when instantiating
        check(
            r#"
struct Foo { field_a: u32 }

fn main() {
    let foo = Foo { field_a<|>: 0, };
}
"#,
            expect![[r#"
                *field_a*

                ```rust
                test::Foo
                ```

                ```rust
                field_a: u32
                ```
            "#]],
        );

        // Hovering over the field in the definition
        check(
            r#"
struct Foo { field_a<|>: u32 }

fn main() {
    let foo = Foo { field_a: 0 };
}
"#,
            expect![[r#"
                *field_a*

                ```rust
                test::Foo
                ```

                ```rust
                field_a: u32
                ```
            "#]],
        );
    }

    #[test]
    fn hover_const_static() {
        check(
            r#"const foo<|>: u32 = 123;"#,
            expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                const foo: u32 = 123
                ```
            "#]],
        );
        check(
            r#"static foo<|>: u32 = 456;"#,
            expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                static foo: u32
                ```
            "#]],
        );
    }

    #[test]
    fn hover_default_generic_types() {
        check(
            r#"
struct Test<K, T = u8> { k: K, t: T }

fn main() {
    let zz<|> = Test { t: 23u8, k: 33 };
}"#,
            expect![[r#"
                *zz*

                ```rust
                Test<i32, u8>
                ```
            "#]],
        );
    }

    #[test]
    fn hover_some() {
        check(
            r#"
enum Option<T> { Some(T) }
use Option::Some;

fn main() { So<|>me(12); }
"#,
            expect![[r#"
                *Some*

                ```rust
                test::Option
                ```

                ```rust
                Some
                ```
            "#]],
        );

        check(
            r#"
enum Option<T> { Some(T) }
use Option::Some;

fn main() { let b<|>ar = Some(12); }
"#,
            expect![[r#"
                *bar*

                ```rust
                Option<i32>
                ```
            "#]],
        );
    }

    #[test]
    fn hover_enum_variant() {
        check(
            r#"
enum Option<T> {
    /// The None variant
    Non<|>e
}
"#,
            expect![[r#"
                *None*

                ```rust
                test::Option
                ```

                ```rust
                None
                ```

                ---

                The None variant
            "#]],
        );

        check(
            r#"
enum Option<T> {
    /// The Some variant
    Some(T)
}
fn main() {
    let s = Option::Som<|>e(12);
}
"#,
            expect![[r#"
                *Some*

                ```rust
                test::Option
                ```

                ```rust
                Some
                ```

                ---

                The Some variant
            "#]],
        );
    }

    #[test]
    fn hover_for_local_variable() {
        check(
            r#"fn func(foo: i32) { fo<|>o; }"#,
            expect![[r#"
                *foo*

                ```rust
                i32
                ```
            "#]],
        )
    }

    #[test]
    fn hover_for_local_variable_pat() {
        check(
            r#"fn func(fo<|>o: i32) {}"#,
            expect![[r#"
                *foo*

                ```rust
                i32
                ```
            "#]],
        )
    }

    #[test]
    fn hover_local_var_edge() {
        check(
            r#"fn func(foo: i32) { if true { <|>foo; }; }"#,
            expect![[r#"
                *foo*

                ```rust
                i32
                ```
            "#]],
        )
    }

    #[test]
    fn hover_for_param_edge() {
        check(
            r#"fn func(<|>foo: i32) {}"#,
            expect![[r#"
                *foo*

                ```rust
                i32
                ```
            "#]],
        )
    }

    #[test]
    fn hover_for_param_with_multiple_traits() {
        check(
            r#"trait Deref {
                type Target: ?Sized;
            }
            trait DerefMut {
                type Target: ?Sized;
            }
            fn f(_x<|>: impl Deref<Target=u8> + DerefMut<Target=u8>) {}"#,
            expect![[r#"
                *_x*

                ```rust
                impl Deref<Target = u8> + DerefMut<Target = u8>
                ```
            "#]],
        )
    }

    #[test]
    fn test_hover_infer_associated_method_result() {
        check(
            r#"
struct Thing { x: u32 }

impl Thing {
    fn new() -> Thing { Thing { x: 0 } }
}

fn main() { let foo_<|>test = Thing::new(); }
            "#,
            expect![[r#"
                *foo_test*

                ```rust
                Thing
                ```
            "#]],
        )
    }

    #[test]
    fn test_hover_infer_associated_method_exact() {
        check(
            r#"
mod wrapper {
    struct Thing { x: u32 }

    impl Thing {
        fn new() -> Thing { Thing { x: 0 } }
    }
}

fn main() { let foo_test = wrapper::Thing::new<|>(); }
"#,
            expect![[r#"
                *new*

                ```rust
                test::wrapper::Thing
                ```

                ```rust
                fn new() -> Thing
                ```
            "#]],
        )
    }

    #[test]
    fn test_hover_infer_associated_const_in_pattern() {
        check(
            r#"
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
"#,
            expect![[r#"
                *C*

                ```rust
                test
                ```

                ```rust
                const C: u32 = 1
                ```
            "#]],
        )
    }

    #[test]
    fn test_hover_self() {
        check(
            r#"
struct Thing { x: u32 }
impl Thing {
    fn new() -> Self { Self<|> { x: 0 } }
}
"#,
            expect![[r#"
                *Self { x: 0 }*
                ```rust
                Thing
                ```
            "#]],
        )
    } /* FIXME: revive these tests
              let (analysis, position) = analysis_and_position(
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
              assert_eq!(trim_markup(&hover.info.markup.as_str()), ("Thing"));

              let (analysis, position) = analysis_and_position(
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
              assert_eq!(trim_markup(&hover.info.markup.as_str()), ("enum Thing"));

              let (analysis, position) = analysis_and_position(
                  "
                  enum Thing { A }
                  impl Thing {
                      pub fn thing(a: Self<|>) {
                      }
                  }
                  ",
              );
              let hover = analysis.hover(position).unwrap().unwrap();
              assert_eq!(trim_markup(&hover.info.markup.as_str()), ("enum Thing"));
      */

    #[test]
    fn test_hover_shadowing_pat() {
        check(
            r#"
fn x() {}

fn y() {
    let x = 0i32;
    x<|>;
}
"#,
            expect![[r#"
                *x*

                ```rust
                i32
                ```
            "#]],
        )
    }

    #[test]
    fn test_hover_macro_invocation() {
        check(
            r#"
macro_rules! foo { () => {} }

fn f() { fo<|>o!(); }
"#,
            expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                macro_rules! foo
                ```
            "#]],
        )
    }

    #[test]
    fn test_hover_tuple_field() {
        check(
            r#"struct TS(String, i32<|>);"#,
            expect![[r#"
                *i32*
                i32
            "#]],
        )
    }

    #[test]
    fn test_hover_through_macro() {
        check(
            r#"
macro_rules! id { ($($tt:tt)*) => { $($tt)* } }
fn foo() {}
id! {
    fn bar() { fo<|>o(); }
}
"#,
            expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                fn foo()
                ```
            "#]],
        );
    }

    #[test]
    fn test_hover_through_expr_in_macro() {
        check(
            r#"
macro_rules! id { ($($tt:tt)*) => { $($tt)* } }
fn foo(bar:u32) { let a = id!(ba<|>r); }
"#,
            expect![[r#"
                *bar*

                ```rust
                u32
                ```
            "#]],
        );
    }

    #[test]
    fn test_hover_through_expr_in_macro_recursive() {
        check(
            r#"
macro_rules! id_deep { ($($tt:tt)*) => { $($tt)* } }
macro_rules! id { ($($tt:tt)*) => { id_deep!($($tt)*) } }
fn foo(bar:u32) { let a = id!(ba<|>r); }
"#,
            expect![[r#"
                *bar*

                ```rust
                u32
                ```
            "#]],
        );
    }

    #[test]
    fn test_hover_through_func_in_macro_recursive() {
        check(
            r#"
macro_rules! id_deep { ($($tt:tt)*) => { $($tt)* } }
macro_rules! id { ($($tt:tt)*) => { id_deep!($($tt)*) } }
fn bar() -> u32 { 0 }
fn foo() { let a = id!([0u32, bar(<|>)] ); }
"#,
            expect![[r#"
                *bar()*
                ```rust
                u32
                ```
            "#]],
        );
    }

    #[test]
    fn test_hover_through_literal_string_in_macro() {
        check(
            r#"
macro_rules! arr { ($($tt:tt)*) => { [$($tt)*)] } }
fn foo() {
    let mastered_for_itunes = "";
    let _ = arr!("Tr<|>acks", &mastered_for_itunes);
}
"#,
            expect![[r#"
                *"Tracks"*
                ```rust
                &str
                ```
            "#]],
        );
    }

    #[test]
    fn test_hover_through_assert_macro() {
        check(
            r#"
#[rustc_builtin_macro]
macro_rules! assert {}

fn bar() -> bool { true }
fn foo() {
    assert!(ba<|>r());
}
"#,
            expect![[r#"
                *bar*

                ```rust
                test
                ```

                ```rust
                fn bar() -> bool
                ```
            "#]],
        );
    }

    #[test]
    fn test_hover_through_literal_string_in_builtin_macro() {
        check_hover_no_result(
            r#"
            #[rustc_builtin_macro]
            macro_rules! format {}

            fn foo() {
                format!("hel<|>lo {}", 0);
            }
            "#,
        );
    }

    #[test]
    fn test_hover_non_ascii_space_doc() {
        check(
            "
///　<- `\u{3000}` here
fn foo() { }

fn bar() { fo<|>o(); }
",
            expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                fn foo()
                ```

                ---

                \<- `　` here
            "#]],
        );
    }

    #[test]
    fn test_hover_function_show_qualifiers() {
        check(
            r#"async fn foo<|>() {}"#,
            expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                async fn foo()
                ```
            "#]],
        );
        check(
            r#"pub const unsafe fn foo<|>() {}"#,
            expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                pub const unsafe fn foo()
                ```
            "#]],
        );
        check(
            r#"pub(crate) async unsafe extern "C" fn foo<|>() {}"#,
            expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                pub(crate) async unsafe extern "C" fn foo()
                ```
            "#]],
        );
    }

    #[test]
    fn test_hover_trait_show_qualifiers() {
        check_actions(
            r"unsafe trait foo<|>() {}",
            expect![[r#"
                [
                    Implementaion(
                        FilePosition {
                            file_id: FileId(
                                1,
                            ),
                            offset: 13,
                        },
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_extern_crate() {
        check(
            r#"
//- /main.rs
extern crate st<|>d;
//- /std/lib.rs
//! Standard library for this test
//!
//! Printed?
//! abc123
            "#,
            expect![[r#"
            *std*
            Standard library for this test

            Printed?
            abc123
            "#]],
        );
        check(
            r#"
//- /main.rs
extern crate std as ab<|>c;
//- /std/lib.rs
//! Standard library for this test
//!
//! Printed?
//! abc123
            "#,
            expect![[r#"
            *abc*
            Standard library for this test

            Printed?
            abc123
            "#]],
        );
    }

    #[test]
    fn test_hover_mod_with_same_name_as_function() {
        check(
            r#"
use self::m<|>y::Bar;
mod my { pub struct Bar; }

fn my() {}
"#,
            expect![[r#"
                *my*

                ```rust
                test
                ```

                ```rust
                mod my
                ```
            "#]],
        );
    }

    #[test]
    fn test_hover_struct_doc_comment() {
        check(
            r#"
/// bar docs
struct Bar;

fn foo() { let bar = Ba<|>r; }
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                struct Bar
                ```

                ---

                bar docs
            "#]],
        );
    }

    #[test]
    fn test_hover_struct_doc_attr() {
        check(
            r#"
#[doc = "bar docs"]
struct Bar;

fn foo() { let bar = Ba<|>r; }
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                struct Bar
                ```

                ---

                bar docs
            "#]],
        );
    }

    #[test]
    fn test_hover_struct_doc_attr_multiple_and_mixed() {
        check(
            r#"
/// bar docs 0
#[doc = "bar docs 1"]
#[doc = "bar docs 2"]
struct Bar;

fn foo() { let bar = Ba<|>r; }
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                struct Bar
                ```

                ---

                bar docs 0

                bar docs 1

                bar docs 2
            "#]],
        );
    }

    #[test]
    fn test_hover_path_link() {
        check(
            r#"
pub struct Foo;
/// [Foo](struct.Foo.html)
pub struct B<|>ar
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                pub struct Bar
                ```

                ---

                [Foo](https://docs.rs/test/*/test/struct.Foo.html)
            "#]],
        );
    }

    #[test]
    fn test_hover_path_link_no_strip() {
        check(
            r#"
pub struct Foo;
/// [struct Foo](struct.Foo.html)
pub struct B<|>ar
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                pub struct Bar
                ```

                ---

                [struct Foo](https://docs.rs/test/*/test/struct.Foo.html)
            "#]],
        );
    }

    #[ignore = "path based links currently only support documentation on ModuleDef items"]
    #[test]
    fn test_hover_path_link_field() {
        check(
            r#"
pub struct Foo;
pub struct Bar {
    /// [Foo](struct.Foo.html)
    fie<|>ld: ()
}
"#,
            expect![[r#"
                *field*

                ```rust
                test::Bar
                ```

                ```rust
                field: ()
                ```

                ---

                [Foo](https://docs.rs/test/*/test/struct.Foo.html)
            "#]],
        );
    }

    #[test]
    fn test_hover_intra_link() {
        check(
            r#"
pub mod foo {
    pub struct Foo;
}
/// [Foo](foo::Foo)
pub struct B<|>ar
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                pub struct Bar
                ```

                ---

                [Foo](https://docs.rs/test/*/test/foo/struct.Foo.html)
            "#]],
        );
    }

    #[test]
    fn test_hover_intra_link_html_root_url() {
        check(
            r#"
#![doc(arbitrary_attribute = "test", html_root_url = "https:/example.com", arbitrary_attribute2)]

pub mod foo {
    pub struct Foo;
}
/// [Foo](foo::Foo)
pub struct B<|>ar
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                pub struct Bar
                ```

                ---

                [Foo](https://example.com/test/foo/struct.Foo.html)
            "#]],
        );
    }

    #[test]
    fn test_hover_intra_link_shortlink() {
        check(
            r#"
pub struct Foo;
/// [Foo]
pub struct B<|>ar
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                pub struct Bar
                ```

                ---

                [Foo](https://docs.rs/test/*/test/struct.Foo.html)
            "#]],
        );
    }

    #[test]
    fn test_hover_intra_link_shortlink_code() {
        check(
            r#"
pub struct Foo;
/// [`Foo`]
pub struct B<|>ar
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                pub struct Bar
                ```

                ---

                [`Foo`](https://docs.rs/test/*/test/struct.Foo.html)
            "#]],
        );
    }

    #[test]
    fn test_hover_intra_link_namespaced() {
        check(
            r#"
pub struct Foo;
fn Foo() {}
/// [Foo()]
pub struct B<|>ar
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                pub struct Bar
                ```

                ---

                [Foo](https://docs.rs/test/*/test/struct.Foo.html)
            "#]],
        );
    }

    #[test]
    fn test_hover_intra_link_shortlink_namspaced_code() {
        check(
            r#"
pub struct Foo;
/// [`struct Foo`]
pub struct B<|>ar
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                pub struct Bar
                ```

                ---

                [`Foo`](https://docs.rs/test/*/test/struct.Foo.html)
            "#]],
        );
    }

    #[test]
    fn test_hover_intra_link_shortlink_namspaced_code_with_at() {
        check(
            r#"
pub struct Foo;
/// [`struct@Foo`]
pub struct B<|>ar
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                pub struct Bar
                ```

                ---

                [`Foo`](https://docs.rs/test/*/test/struct.Foo.html)
            "#]],
        );
    }

    #[test]
    fn test_hover_intra_link_reference() {
        check(
            r#"
pub struct Foo;
/// [my Foo][foo]
///
/// [foo]: Foo
pub struct B<|>ar
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                pub struct Bar
                ```

                ---

                [my Foo](https://docs.rs/test/*/test/struct.Foo.html)
            "#]],
        );
    }

    #[test]
    fn test_hover_external_url() {
        check(
            r#"
pub struct Foo;
/// [external](https://www.google.com)
pub struct B<|>ar
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                pub struct Bar
                ```

                ---

                [external](https://www.google.com)
            "#]],
        );
    }

    // Check that we don't rewrite links which we can't identify
    #[test]
    fn test_hover_unknown_target() {
        check(
            r#"
pub struct Foo;
/// [baz](Baz)
pub struct B<|>ar
"#,
            expect![[r#"
                *Bar*

                ```rust
                test
                ```

                ```rust
                pub struct Bar
                ```

                ---

                [baz](Baz)
            "#]],
        );
    }

    #[test]
    fn test_doc_links_enum_variant() {
        check(
            r#"
enum E {
    /// [E]
    V<|> { field: i32 }
}
"#,
            expect![[r#"
                *V*

                ```rust
                test::E
                ```

                ```rust
                V
                ```

                ---

                [E](https://docs.rs/test/*/test/enum.E.html)
            "#]],
        );
    }

    #[test]
    fn test_doc_links_field() {
        check(
            r#"
struct S {
    /// [`S`]
    field<|>: i32
}
"#,
            expect![[r#"
                *field*

                ```rust
                test::S
                ```

                ```rust
                field: i32
                ```

                ---

                [`S`](https://docs.rs/test/*/test/struct.S.html)
            "#]],
        );
    }

    #[test]
    fn test_hover_macro_generated_struct_fn_doc_comment() {
        mark::check!(hover_macro_generated_struct_fn_doc_comment);

        check(
            r#"
macro_rules! bar {
    () => {
        struct Bar;
        impl Bar {
            /// Do the foo
            fn foo(&self) {}
        }
    }
}

bar!();

fn foo() { let bar = Bar; bar.fo<|>o(); }
"#,
            expect![[r#"
                *foo*

                ```rust
                test::Bar
                ```

                ```rust
                fn foo(&self)
                ```

                ---

                Do the foo
            "#]],
        );
    }

    #[test]
    fn test_hover_macro_generated_struct_fn_doc_attr() {
        mark::check!(hover_macro_generated_struct_fn_doc_attr);

        check(
            r#"
macro_rules! bar {
    () => {
        struct Bar;
        impl Bar {
            #[doc = "Do the foo"]
            fn foo(&self) {}
        }
    }
}

bar!();

fn foo() { let bar = Bar; bar.fo<|>o(); }
"#,
            expect![[r#"
                *foo*

                ```rust
                test::Bar
                ```

                ```rust
                fn foo(&self)
                ```

                ---

                Do the foo
            "#]],
        );
    }

    #[test]
    fn test_hover_trait_has_impl_action() {
        check_actions(
            r#"trait foo<|>() {}"#,
            expect![[r#"
                [
                    Implementaion(
                        FilePosition {
                            file_id: FileId(
                                1,
                            ),
                            offset: 6,
                        },
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_struct_has_impl_action() {
        check_actions(
            r"struct foo<|>() {}",
            expect![[r#"
                [
                    Implementaion(
                        FilePosition {
                            file_id: FileId(
                                1,
                            ),
                            offset: 7,
                        },
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_union_has_impl_action() {
        check_actions(
            r#"union foo<|>() {}"#,
            expect![[r#"
                [
                    Implementaion(
                        FilePosition {
                            file_id: FileId(
                                1,
                            ),
                            offset: 6,
                        },
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_enum_has_impl_action() {
        check_actions(
            r"enum foo<|>() { A, B }",
            expect![[r#"
                [
                    Implementaion(
                        FilePosition {
                            file_id: FileId(
                                1,
                            ),
                            offset: 5,
                        },
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_test_has_action() {
        check_actions(
            r#"
#[test]
fn foo_<|>test() {}
"#,
            expect![[r#"
                [
                    Runnable(
                        Runnable {
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..24,
                                focus_range: Some(
                                    11..19,
                                ),
                                name: "foo_test",
                                kind: FN,
                                container_name: None,
                                description: None,
                                docs: None,
                            },
                            kind: Test {
                                test_id: Path(
                                    "foo_test",
                                ),
                                attr: TestAttr {
                                    ignore: false,
                                },
                            },
                            cfg_exprs: [],
                        },
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_test_mod_has_action() {
        check_actions(
            r#"
mod tests<|> {
    #[test]
    fn foo_test() {}
}
"#,
            expect![[r#"
                [
                    Runnable(
                        Runnable {
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 0..46,
                                focus_range: Some(
                                    4..9,
                                ),
                                name: "tests",
                                kind: MODULE,
                                container_name: None,
                                description: None,
                                docs: None,
                            },
                            kind: TestMod {
                                path: "tests",
                            },
                            cfg_exprs: [],
                        },
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_struct_has_goto_type_action() {
        check_actions(
            r#"
struct S{ f1: u32 }

fn main() { let s<|>t = S{ f1:0 }; }
            "#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..19,
                                    focus_range: Some(
                                        7..8,
                                    ),
                                    name: "S",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct S",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_generic_struct_has_goto_type_actions() {
        check_actions(
            r#"
struct Arg(u32);
struct S<T>{ f1: T }

fn main() { let s<|>t = S{ f1:Arg(0) }; }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 17..37,
                                    focus_range: Some(
                                        24..25,
                                    ),
                                    name: "S",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct S",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::Arg",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..16,
                                    focus_range: Some(
                                        7..10,
                                    ),
                                    name: "Arg",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct Arg",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_generic_struct_has_flattened_goto_type_actions() {
        check_actions(
            r#"
struct Arg(u32);
struct S<T>{ f1: T }

fn main() { let s<|>t = S{ f1: S{ f1: Arg(0) } }; }
            "#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 17..37,
                                    focus_range: Some(
                                        24..25,
                                    ),
                                    name: "S",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct S",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::Arg",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..16,
                                    focus_range: Some(
                                        7..10,
                                    ),
                                    name: "Arg",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct Arg",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_tuple_has_goto_type_actions() {
        check_actions(
            r#"
struct A(u32);
struct B(u32);
mod M {
    pub struct C(u32);
}

fn main() { let s<|>t = (A(1), B(2), M::C(3) ); }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::A",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..14,
                                    focus_range: Some(
                                        7..8,
                                    ),
                                    name: "A",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct A",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::B",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 15..29,
                                    focus_range: Some(
                                        22..23,
                                    ),
                                    name: "B",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct B",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::M::C",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 42..60,
                                    focus_range: Some(
                                        53..54,
                                    ),
                                    name: "C",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "pub struct C",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_return_impl_trait_has_goto_type_action() {
        check_actions(
            r#"
trait Foo {}
fn foo() -> impl Foo {}

fn main() { let s<|>t = foo(); }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..12,
                                    focus_range: Some(
                                        6..9,
                                    ),
                                    name: "Foo",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Foo",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_generic_return_impl_trait_has_goto_type_action() {
        check_actions(
            r#"
trait Foo<T> {}
struct S;
fn foo() -> impl Foo<S> {}

fn main() { let s<|>t = foo(); }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..15,
                                    focus_range: Some(
                                        6..9,
                                    ),
                                    name: "Foo",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Foo",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 16..25,
                                    focus_range: Some(
                                        23..24,
                                    ),
                                    name: "S",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct S",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_return_impl_traits_has_goto_type_action() {
        check_actions(
            r#"
trait Foo {}
trait Bar {}
fn foo() -> impl Foo + Bar {}

fn main() { let s<|>t = foo(); }
            "#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..12,
                                    focus_range: Some(
                                        6..9,
                                    ),
                                    name: "Foo",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Foo",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::Bar",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 13..25,
                                    focus_range: Some(
                                        19..22,
                                    ),
                                    name: "Bar",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Bar",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_generic_return_impl_traits_has_goto_type_action() {
        check_actions(
            r#"
trait Foo<T> {}
trait Bar<T> {}
struct S1 {}
struct S2 {}

fn foo() -> impl Foo<S1> + Bar<S2> {}

fn main() { let s<|>t = foo(); }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..15,
                                    focus_range: Some(
                                        6..9,
                                    ),
                                    name: "Foo",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Foo",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::Bar",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 16..31,
                                    focus_range: Some(
                                        22..25,
                                    ),
                                    name: "Bar",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Bar",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S1",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 32..44,
                                    focus_range: Some(
                                        39..41,
                                    ),
                                    name: "S1",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct S1",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S2",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 45..57,
                                    focus_range: Some(
                                        52..54,
                                    ),
                                    name: "S2",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct S2",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_arg_impl_trait_has_goto_type_action() {
        check_actions(
            r#"
trait Foo {}
fn foo(ar<|>g: &impl Foo) {}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..12,
                                    focus_range: Some(
                                        6..9,
                                    ),
                                    name: "Foo",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Foo",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_arg_impl_traits_has_goto_type_action() {
        check_actions(
            r#"
trait Foo {}
trait Bar<T> {}
struct S{}

fn foo(ar<|>g: &impl Foo + Bar<S>) {}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..12,
                                    focus_range: Some(
                                        6..9,
                                    ),
                                    name: "Foo",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Foo",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::Bar",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 13..28,
                                    focus_range: Some(
                                        19..22,
                                    ),
                                    name: "Bar",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Bar",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 29..39,
                                    focus_range: Some(
                                        36..37,
                                    ),
                                    name: "S",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct S",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_async_block_impl_trait_has_goto_type_action() {
        check_actions(
            r#"
struct S;
fn foo() {
    let fo<|>o = async { S };
}

#[prelude_import] use future::*;
mod future {
    #[lang = "future_trait"]
    pub trait Future { type Output; }
}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::future::Future",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 101..163,
                                    focus_range: Some(
                                        140..146,
                                    ),
                                    name: "Future",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "pub trait Future",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..9,
                                    focus_range: Some(
                                        7..8,
                                    ),
                                    name: "S",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct S",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_arg_generic_impl_trait_has_goto_type_action() {
        check_actions(
            r#"
trait Foo<T> {}
struct S {}
fn foo(ar<|>g: &impl Foo<S>) {}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..15,
                                    focus_range: Some(
                                        6..9,
                                    ),
                                    name: "Foo",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Foo",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 16..27,
                                    focus_range: Some(
                                        23..24,
                                    ),
                                    name: "S",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct S",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_dyn_return_has_goto_type_action() {
        check_actions(
            r#"
trait Foo {}
struct S;
impl Foo for S {}

struct B<T>{}
fn foo() -> B<dyn Foo> {}

fn main() { let s<|>t = foo(); }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::B",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 42..55,
                                    focus_range: Some(
                                        49..50,
                                    ),
                                    name: "B",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct B",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..12,
                                    focus_range: Some(
                                        6..9,
                                    ),
                                    name: "Foo",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Foo",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_dyn_arg_has_goto_type_action() {
        check_actions(
            r#"
trait Foo {}
fn foo(ar<|>g: &dyn Foo) {}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..12,
                                    focus_range: Some(
                                        6..9,
                                    ),
                                    name: "Foo",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Foo",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_generic_dyn_arg_has_goto_type_action() {
        check_actions(
            r#"
trait Foo<T> {}
struct S {}
fn foo(ar<|>g: &dyn Foo<S>) {}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..15,
                                    focus_range: Some(
                                        6..9,
                                    ),
                                    name: "Foo",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Foo",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 16..27,
                                    focus_range: Some(
                                        23..24,
                                    ),
                                    name: "S",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct S",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_goto_type_action_links_order() {
        check_actions(
            r#"
trait ImplTrait<T> {}
trait DynTrait<T> {}
struct B<T> {}
struct S {}

fn foo(a<|>rg: &impl ImplTrait<B<dyn DynTrait<B<S>>>>) {}
            "#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::ImplTrait",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..21,
                                    focus_range: Some(
                                        6..15,
                                    ),
                                    name: "ImplTrait",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait ImplTrait",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::B",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 43..57,
                                    focus_range: Some(
                                        50..51,
                                    ),
                                    name: "B",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct B",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::DynTrait",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 22..42,
                                    focus_range: Some(
                                        28..36,
                                    ),
                                    name: "DynTrait",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait DynTrait",
                                    ),
                                    docs: None,
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 58..69,
                                    focus_range: Some(
                                        65..66,
                                    ),
                                    name: "S",
                                    kind: STRUCT,
                                    container_name: None,
                                    description: Some(
                                        "struct S",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_associated_type_has_goto_type_action() {
        check_actions(
            r#"
trait Foo {
    type Item;
    fn get(self) -> Self::Item {}
}

struct Bar{}
struct S{}

impl Foo for S { type Item = Bar; }

fn test() -> impl Foo { S {} }

fn main() { let s<|>t = test().get(); }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        1,
                                    ),
                                    full_range: 0..62,
                                    focus_range: Some(
                                        6..9,
                                    ),
                                    name: "Foo",
                                    kind: TRAIT,
                                    container_name: None,
                                    description: Some(
                                        "trait Foo",
                                    ),
                                    docs: None,
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }
}
