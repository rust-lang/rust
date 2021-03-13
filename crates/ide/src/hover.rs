use either::Either;
use hir::{
    Adt, AsAssocItem, AssocItemContainer, FieldSource, GenericParam, HasAttrs, HasSource,
    HirDisplay, Module, ModuleDef, ModuleSource, Semantics,
};
use ide_db::{
    base_db::SourceDatabase,
    defs::{Definition, NameClass, NameRefClass},
    helpers::FamousDefs,
    RootDatabase,
};
use itertools::Itertools;
use stdx::format_to;
use syntax::{ast, match_ast, AstNode, SyntaxKind::*, SyntaxToken, TokenAtOffset, T};

use crate::{
    display::{macro_label, ShortLabel, TryToNav},
    doc_links::{remove_links, rewrite_links},
    markdown_remove::remove_markdown,
    markup::Markup,
    runnables::{runnable_fn, runnable_mod},
    FileId, FilePosition, NavigationTarget, RangeInfo, Runnable,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HoverConfig {
    pub implementations: bool,
    pub run: bool,
    pub debug: bool,
    pub goto_type_def: bool,
    pub links_in_hover: bool,
    pub markdown: bool,
}

impl HoverConfig {
    pub const NO_ACTIONS: Self = Self {
        implementations: false,
        run: false,
        debug: false,
        goto_type_def: false,
        links_in_hover: true,
        markdown: true,
    };

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
    Implementation(FilePosition),
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
pub(crate) fn hover(
    db: &RootDatabase,
    position: FilePosition,
    links_in_hover: bool,
    markdown: bool,
) -> Option<RangeInfo<HoverResult>> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id).syntax().clone();
    let token = pick_best(file.token_at_offset(position.offset))?;
    let token = sema.descend_into_macros(token);

    let mut res = HoverResult::default();

    let node = token.parent();
    let definition = match_ast! {
        match node {
            // we don't use NameClass::referenced_or_defined here as we do not want to resolve
            // field pattern shorthands to their definition
            ast::Name(name) => NameClass::classify(&sema, &name).and_then(|class| match class {
                NameClass::ConstReference(def) => Some(def),
                def => def.defined(sema.db),
            }),
            ast::NameRef(name_ref) => NameRefClass::classify(&sema, &name_ref).map(|d| d.referenced(sema.db)),
            ast::Lifetime(lifetime) => NameClass::classify_lifetime(&sema, &lifetime)
                .map_or_else(|| NameRefClass::classify_lifetime(&sema, &lifetime).map(|d| d.referenced(sema.db)), |d| d.defined(sema.db)),
            _ => None,
        }
    };
    if let Some(definition) = definition {
        let famous_defs = match &definition {
            Definition::ModuleDef(ModuleDef::BuiltinType(_)) => {
                Some(FamousDefs(&sema, sema.scope(&node).krate()))
            }
            _ => None,
        };
        if let Some(markup) = hover_for_definition(db, definition, famous_defs.as_ref()) {
            res.markup = process_markup(sema.db, definition, &markup, links_in_hover, markdown);
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

    if token.kind() == syntax::SyntaxKind::COMMENT {
        // don't highlight the entire parent node on comment hover
        return None;
    }
    if let res @ Some(_) = hover_for_keyword(&sema, links_in_hover, markdown, &token) {
        return res;
    }

    let node = token
        .ancestors()
        .find(|n| ast::Expr::can_cast(n.kind()) || ast::Pat::can_cast(n.kind()))?;

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

    res.markup = if markdown {
        Markup::fenced_block(&ty.display(db))
    } else {
        ty.display(db).to_string().into()
    };
    let range = sema.original_range(&node).range;
    Some(RangeInfo::new(range, res))
}

fn show_implementations_action(db: &RootDatabase, def: Definition) -> Option<HoverAction> {
    fn to_action(nav_target: NavigationTarget) -> HoverAction {
        HoverAction::Implementation(FilePosition {
            file_id: nav_target.file_id,
            offset: nav_target.focus_or_full_range().start(),
        })
    }

    let adt = match def {
        Definition::ModuleDef(ModuleDef::Trait(it)) => return it.try_to_nav(db).map(to_action),
        Definition::ModuleDef(ModuleDef::Adt(it)) => Some(it),
        Definition::SelfType(it) => it.target_ty(db).as_adt(),
        _ => None,
    }?;
    adt.try_to_nav(db).map(to_action)
}

fn runnable_action(
    sema: &Semantics<RootDatabase>,
    def: Definition,
    file_id: FileId,
) -> Option<HoverAction> {
    match def {
        Definition::ModuleDef(it) => match it {
            ModuleDef::Module(it) => runnable_mod(&sema, it).map(|it| HoverAction::Runnable(it)),
            ModuleDef::Function(func) => {
                let src = func.source(sema.db)?;
                if src.file_id != file_id.into() {
                    cov_mark::hit!(hover_macro_generated_struct_fn_doc_comment);
                    cov_mark::hit!(hover_macro_generated_struct_fn_doc_attr);
                    return None;
                }

                runnable_fn(&sema, func).map(HoverAction::Runnable)
            }
            _ => None,
        },
        _ => None,
    }
}

fn goto_type_action(db: &RootDatabase, def: Definition) -> Option<HoverAction> {
    let mut targets: Vec<ModuleDef> = Vec::new();
    let mut push_new_def = |item: ModuleDef| {
        if !targets.contains(&item) {
            targets.push(item);
        }
    };

    if let Definition::GenericParam(GenericParam::TypeParam(it)) = def {
        it.trait_bounds(db).into_iter().for_each(|it| push_new_def(it.into()));
    } else {
        let ty = match def {
            Definition::Local(it) => it.ty(db),
            Definition::GenericParam(GenericParam::ConstParam(it)) => it.ty(db),
            _ => return None,
        };

        ty.walk(db, |t| {
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
    }

    let targets = targets
        .into_iter()
        .filter_map(|it| {
            Some(HoverGotoTypeData {
                mod_path: render_path(db, it.module(db)?, it.name(db).map(|name| name.to_string())),
                nav: it.try_to_nav(db)?,
            })
        })
        .collect();

    Some(HoverAction::GoToType(targets))
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

fn process_markup(
    db: &RootDatabase,
    def: Definition,
    markup: &Markup,
    links_in_hover: bool,
    markdown: bool,
) -> Markup {
    let markup = markup.as_str();
    let markup = if !markdown {
        remove_markdown(markup)
    } else if links_in_hover {
        rewrite_links(db, markup, &def)
    } else {
        remove_links(markup)
    };
    Markup::from(markup)
}

fn definition_owner_name(db: &RootDatabase, def: &Definition) -> Option<String> {
    match def {
        Definition::Field(f) => Some(f.parent_def(db).name(db)),
        Definition::Local(l) => l.parent(db).name(db),
        Definition::ModuleDef(md) => match md {
            ModuleDef::Function(f) => match f.as_assoc_item(db)?.container(db) {
                AssocItemContainer::Trait(t) => Some(t.name(db)),
                AssocItemContainer::Impl(i) => i.target_ty(db).as_adt().map(|adt| adt.name(db)),
            },
            ModuleDef::Variant(e) => Some(e.parent_enum(db).name(db)),
            _ => None,
        },
        _ => None,
    }
    .map(|name| name.to_string())
}

fn render_path(db: &RootDatabase, module: Module, item_name: Option<String>) -> String {
    let crate_name =
        db.crate_graph()[module.krate().into()].display_name.as_ref().map(|it| it.to_string());
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

fn hover_for_definition(
    db: &RootDatabase,
    def: Definition,
    famous_defs: Option<&FamousDefs>,
) -> Option<Markup> {
    let mod_path = definition_mod_path(db, &def);
    return match def {
        Definition::Macro(it) => {
            let label = macro_label(&it.source(db)?.value);
            from_def_source_labeled(db, it, Some(label), mod_path)
        }
        Definition::Field(def) => {
            let src = def.source(db)?.value;
            if let FieldSource::Named(it) = src {
                from_def_source_labeled(db, def, it.short_label(), mod_path)
            } else {
                None
            }
        }
        Definition::ModuleDef(it) => match it {
            ModuleDef::Module(it) => from_def_source_labeled(
                db,
                it,
                match it.definition_source(db).value {
                    ModuleSource::Module(it) => it.short_label(),
                    ModuleSource::SourceFile(it) => it.short_label(),
                    ModuleSource::BlockExpr(it) => it.short_label(),
                },
                mod_path,
            ),
            ModuleDef::Function(it) => from_def_source(db, it, mod_path),
            ModuleDef::Adt(Adt::Struct(it)) => from_def_source(db, it, mod_path),
            ModuleDef::Adt(Adt::Union(it)) => from_def_source(db, it, mod_path),
            ModuleDef::Adt(Adt::Enum(it)) => from_def_source(db, it, mod_path),
            ModuleDef::Variant(it) => from_def_source(db, it, mod_path),
            ModuleDef::Const(it) => from_def_source(db, it, mod_path),
            ModuleDef::Static(it) => from_def_source(db, it, mod_path),
            ModuleDef::Trait(it) => from_def_source(db, it, mod_path),
            ModuleDef::TypeAlias(it) => from_def_source(db, it, mod_path),
            ModuleDef::BuiltinType(it) => famous_defs
                .and_then(|fd| hover_for_builtin(fd, it))
                .or_else(|| Some(Markup::fenced_block(&it.name()))),
        },
        Definition::Local(it) => hover_for_local(it, db),
        Definition::SelfType(impl_def) => {
            impl_def.target_ty(db).as_adt().and_then(|adt| match adt {
                Adt::Struct(it) => from_def_source(db, it, mod_path),
                Adt::Union(it) => from_def_source(db, it, mod_path),
                Adt::Enum(it) => from_def_source(db, it, mod_path),
            })
        }
        Definition::Label(it) => Some(Markup::fenced_block(&it.name(db))),
        Definition::GenericParam(it) => match it {
            GenericParam::TypeParam(it) => Some(Markup::fenced_block(&it.display(db))),
            GenericParam::LifetimeParam(it) => Some(Markup::fenced_block(&it.name(db))),
            GenericParam::ConstParam(it) => from_def_source(db, it, None),
        },
    };

    fn from_def_source<A, D>(db: &RootDatabase, def: D, mod_path: Option<String>) -> Option<Markup>
    where
        D: HasSource<Ast = A> + HasAttrs + Copy,
        A: ShortLabel,
    {
        let short_label = def.source(db)?.value.short_label();
        from_def_source_labeled(db, def, short_label, mod_path)
    }

    fn from_def_source_labeled<D>(
        db: &RootDatabase,
        def: D,
        short_label: Option<String>,
        mod_path: Option<String>,
    ) -> Option<Markup>
    where
        D: HasAttrs,
    {
        let docs = def.attrs(db).docs().map(Into::into);
        hover_markup(docs, short_label, mod_path)
    }
}

fn hover_for_local(it: hir::Local, db: &RootDatabase) -> Option<Markup> {
    let ty = it.ty(db);
    let ty = ty.display(db);
    let is_mut = if it.is_mut(db) { "mut " } else { "" };
    let desc = match it.source(db).value {
        Either::Left(ident) => {
            let name = it.name(db).unwrap();
            let let_kw = if ident
                .syntax()
                .parent()
                .map_or(false, |p| p.kind() == LET_STMT || p.kind() == CONDITION)
            {
                "let "
            } else {
                ""
            };
            format!("{}{}{}: {}", let_kw, is_mut, name, ty)
        }
        Either::Right(_) => format!("{}self: {}", is_mut, ty),
    };
    hover_markup(None, Some(desc), None)
}

fn hover_for_keyword(
    sema: &Semantics<RootDatabase>,
    links_in_hover: bool,
    markdown: bool,
    token: &SyntaxToken,
) -> Option<RangeInfo<HoverResult>> {
    if !token.kind().is_keyword() {
        return None;
    }
    let famous_defs = FamousDefs(&sema, sema.scope(&token.parent()).krate());
    // std exposes {}_keyword modules with docstrings on the root to document keywords
    let keyword_mod = format!("{}_keyword", token.text());
    let doc_owner = find_std_module(&famous_defs, &keyword_mod)?;
    let docs = doc_owner.attrs(sema.db).docs()?;
    let markup = process_markup(
        sema.db,
        Definition::ModuleDef(doc_owner.into()),
        &hover_markup(Some(docs.into()), Some(token.text().into()), None)?,
        links_in_hover,
        markdown,
    );
    Some(RangeInfo::new(token.text_range(), HoverResult { markup, actions: Default::default() }))
}

fn hover_for_builtin(famous_defs: &FamousDefs, builtin: hir::BuiltinType) -> Option<Markup> {
    // std exposes prim_{} modules with docstrings on the root to document the builtins
    let primitive_mod = format!("prim_{}", builtin.name());
    let doc_owner = find_std_module(famous_defs, &primitive_mod)?;
    let docs = doc_owner.attrs(famous_defs.0.db).docs()?;
    hover_markup(Some(docs.into()), Some(builtin.name().to_string()), None)
}

fn find_std_module(famous_defs: &FamousDefs, name: &str) -> Option<hir::Module> {
    let db = famous_defs.0.db;
    let std_crate = famous_defs.std()?;
    let std_root_module = std_crate.root_module(db);
    std_root_module
        .children(db)
        .find(|module| module.name(db).map_or(false, |module| module.to_string() == name))
}

fn pick_best(tokens: TokenAtOffset<SyntaxToken>) -> Option<SyntaxToken> {
    return tokens.max_by_key(priority);
    fn priority(n: &SyntaxToken) -> usize {
        match n.kind() {
            IDENT | INT_NUMBER | LIFETIME_IDENT | T![self] | T![super] | T![crate] => 3,
            T!['('] | T![')'] => 2,
            kind if kind.is_trivia() => 0,
            _ => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use ide_db::base_db::FileLoader;

    use crate::fixture;

    use super::*;

    fn check_hover_no_result(ra_fixture: &str) {
        let (analysis, position) = fixture::position(ra_fixture);
        assert!(analysis.hover(position, true, true).unwrap().is_none());
    }

    fn check(ra_fixture: &str, expect: Expect) {
        let (analysis, position) = fixture::position(ra_fixture);
        let hover = analysis.hover(position, true, true).unwrap().unwrap();

        let content = analysis.db.file_text(position.file_id);
        let hovered_element = &content[hover.range];

        let actual = format!("*{}*\n{}\n", hovered_element, hover.info.markup);
        expect.assert_eq(&actual)
    }

    fn check_hover_no_links(ra_fixture: &str, expect: Expect) {
        let (analysis, position) = fixture::position(ra_fixture);
        let hover = analysis.hover(position, false, true).unwrap().unwrap();

        let content = analysis.db.file_text(position.file_id);
        let hovered_element = &content[hover.range];

        let actual = format!("*{}*\n{}\n", hovered_element, hover.info.markup);
        expect.assert_eq(&actual)
    }

    fn check_hover_no_markdown(ra_fixture: &str, expect: Expect) {
        let (analysis, position) = fixture::position(ra_fixture);
        let hover = analysis.hover(position, true, false).unwrap().unwrap();

        let content = analysis.db.file_text(position.file_id);
        let hovered_element = &content[hover.range];

        let actual = format!("*{}*\n{}\n", hovered_element, hover.info.markup);
        expect.assert_eq(&actual)
    }

    fn check_actions(ra_fixture: &str, expect: Expect) {
        let (analysis, position) = fixture::position(ra_fixture);
        let hover = analysis.hover(position, true, true).unwrap().unwrap();
        expect.assert_debug_eq(&hover.info.actions)
    }

    #[test]
    fn hover_shows_type_of_an_expression() {
        check(
            r#"
pub fn foo() -> u32 { 1 }

fn main() {
    let foo_test = foo()$0;
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
    fn hover_remove_markdown_if_configured() {
        check_hover_no_markdown(
            r#"
pub fn foo() -> u32 { 1 }

fn main() {
    let foo_test = foo()$0;
}
"#,
            expect![[r#"
                *foo()*
                u32
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
    let mut iter$0 = scan(OtherStruct { i: num }, closure, number);
}
"#,
            expect![[r#"
                *iter*

                ```rust
                let mut iter: Iter<Scan<OtherStruct<OtherStruct<i32>>, |&mut u32, &u32, &mut u32| -> Option<u32>, u32>>
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

fn main() { let foo_test = fo$0o(); }
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

fn main() { let foo_test = fo$0o(); }
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

fn main() { let foo_test = fo$0o(); }
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
pub fn foo$0(a: u32, b: u32) -> u32 {}

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
pub fn foo$0(_: &Path) {}

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
    fn hover_shows_fn_doc_attr_raw_string() {
        check(
            r##"
#[doc = r#"Raw string doc attr"#]
pub fn foo$0(_: &Path) {}

fn main() { }
"##,
            expect![[r##"
                *foo*

                ```rust
                test
                ```

                ```rust
                pub fn foo(_: &Path)
                ```

                ---

                Raw string doc attr
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
    let foo = Foo { field_a$0: 0, };
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
struct Foo { field_a$0: u32 }

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
            r#"const foo$0: u32 = 123;"#,
            expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                const foo: u32
                ```
            "#]],
        );
        check(
            r#"static foo$0: u32 = 456;"#,
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
    let zz$0 = Test { t: 23u8, k: 33 };
}"#,
            expect![[r#"
                *zz*

                ```rust
                let zz: Test<i32, u8>
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

fn main() { So$0me(12); }
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

fn main() { let b$0ar = Some(12); }
"#,
            expect![[r#"
                *bar*

                ```rust
                let bar: Option<i32>
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
    Non$0e
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
    let s = Option::Som$0e(12);
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
            r#"fn func(foo: i32) { fo$0o; }"#,
            expect![[r#"
                *foo*

                ```rust
                foo: i32
                ```
            "#]],
        )
    }

    #[test]
    fn hover_for_local_variable_pat() {
        check(
            r#"fn func(fo$0o: i32) {}"#,
            expect![[r#"
                *foo*

                ```rust
                foo: i32
                ```
            "#]],
        )
    }

    #[test]
    fn hover_local_var_edge() {
        check(
            r#"fn func(foo: i32) { if true { $0foo; }; }"#,
            expect![[r#"
                *foo*

                ```rust
                foo: i32
                ```
            "#]],
        )
    }

    #[test]
    fn hover_for_param_edge() {
        check(
            r#"fn func($0foo: i32) {}"#,
            expect![[r#"
                *foo*

                ```rust
                foo: i32
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
            fn f(_x$0: impl Deref<Target=u8> + DerefMut<Target=u8>) {}"#,
            expect![[r#"
                *_x*

                ```rust
                _x: impl Deref<Target = u8> + DerefMut<Target = u8>
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

fn main() { let foo_$0test = Thing::new(); }
            "#,
            expect![[r#"
                *foo_test*

                ```rust
                let foo_test: Thing
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

fn main() { let foo_test = wrapper::Thing::new$0(); }
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
        X::C$0 => {},
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
                const C: u32
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
    fn new() -> Self { Self$0 { x: 0 } }
}
"#,
            expect![[r#"
                *Self*

                ```rust
                test
                ```

                ```rust
                struct Thing
                ```
            "#]],
        );
        check(
            r#"
struct Thing { x: u32 }
impl Thing {
    fn new() -> Self$0 { Self { x: 0 } }
}
"#,
            expect![[r#"
                *Self*

                ```rust
                test
                ```

                ```rust
                struct Thing
                ```
            "#]],
        );
        check(
            r#"
enum Thing { A }
impl Thing {
    pub fn new() -> Self$0 { Thing::A }
}
"#,
            expect![[r#"
                *Self*

                ```rust
                test
                ```

                ```rust
                enum Thing
                ```
            "#]],
        );
        check(
            r#"
        enum Thing { A }
        impl Thing {
            pub fn thing(a: Self$0) {}
        }
        "#,
            expect![[r#"
                *Self*

                ```rust
                test
                ```

                ```rust
                enum Thing
                ```
            "#]],
        );
    }

    #[test]
    fn test_hover_shadowing_pat() {
        check(
            r#"
fn x() {}

fn y() {
    let x = 0i32;
    x$0;
}
"#,
            expect![[r#"
                *x*

                ```rust
                let x: i32
                ```
            "#]],
        )
    }

    #[test]
    fn test_hover_macro_invocation() {
        check(
            r#"
macro_rules! foo { () => {} }

fn f() { fo$0o!(); }
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
            r#"struct TS(String, i32$0);"#,
            expect![[r#"
                *i32*

                ```rust
                i32
                ```
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
    fn bar() { fo$0o(); }
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
fn foo(bar:u32) { let a = id!(ba$0r); }
"#,
            expect![[r#"
                *bar*

                ```rust
                bar: u32
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
fn foo(bar:u32) { let a = id!(ba$0r); }
"#,
            expect![[r#"
                *bar*

                ```rust
                bar: u32
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
fn foo() { let a = id!([0u32, bar($0)] ); }
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
    let _ = arr!("Tr$0acks", &mastered_for_itunes);
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
    assert!(ba$0r());
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
                format!("hel$0lo {}", 0);
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

fn bar() { fo$0o(); }
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
            r#"async fn foo$0() {}"#,
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
            r#"pub const unsafe fn foo$0() {}"#,
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
            r#"pub(crate) async unsafe extern "C" fn foo$0() {}"#,
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
            r"unsafe trait foo$0() {}",
            expect![[r#"
                [
                    Implementation(
                        FilePosition {
                            file_id: FileId(
                                0,
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
//- /main.rs crate:main deps:std
extern crate st$0d;
//- /std/lib.rs crate:std
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
//- /main.rs crate:main deps:std
extern crate std as ab$0c;
//- /std/lib.rs crate:std
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
use self::m$0y::Bar;
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

fn foo() { let bar = Ba$0r; }
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

fn foo() { let bar = Ba$0r; }
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

fn foo() { let bar = Ba$0r; }
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
pub struct B$0ar
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
pub struct B$0ar
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
    fie$0ld: ()
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
pub struct B$0ar
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
pub struct B$0ar
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
pub struct B$0ar
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
pub struct B$0ar
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
pub struct B$0ar
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
pub struct B$0ar
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
pub struct B$0ar
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
pub struct B$0ar
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
    fn test_hover_intra_link_reference_to_trait_method() {
        check(
            r#"
pub trait Foo {
    fn buzz() -> usize;
}
/// [Foo][buzz]
///
/// [buzz]: Foo::buzz
pub struct B$0ar
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

                [Foo](https://docs.rs/test/*/test/trait.Foo.html#tymethod.buzz)
            "#]],
        );
    }

    #[test]
    fn test_hover_external_url() {
        check(
            r#"
pub struct Foo;
/// [external](https://www.google.com)
pub struct B$0ar
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
pub struct B$0ar
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
    V$0 { field: i32 }
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
    field$0: i32
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
    fn test_hover_no_links() {
        check_hover_no_links(
            r#"
/// Test cases:
/// case 1.  bare URL: https://www.example.com/
/// case 2.  inline URL with title: [example](https://www.example.com/)
/// case 3.  code reference: [`Result`]
/// case 4.  code reference but miss footnote: [`String`]
/// case 5.  autolink: <http://www.example.com/>
/// case 6.  email address: <test@example.com>
/// case 7.  reference: [example][example]
/// case 8.  collapsed link: [example][]
/// case 9.  shortcut link: [example]
/// case 10. inline without URL: [example]()
/// case 11. reference: [foo][foo]
/// case 12. reference: [foo][bar]
/// case 13. collapsed link: [foo][]
/// case 14. shortcut link: [foo]
/// case 15. inline without URL: [foo]()
/// case 16. just escaped text: \[foo]
/// case 17. inline link: [Foo](foo::Foo)
///
/// [`Result`]: ../../std/result/enum.Result.html
/// [^example]: https://www.example.com/
pub fn fo$0o() {}
"#,
            expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                pub fn foo()
                ```

                ---

                Test cases:
                case 1.  bare URL: https://www.example.com/
                case 2.  inline URL with title: [example](https://www.example.com/)
                case 3.  code reference: `Result`
                case 4.  code reference but miss footnote: `String`
                case 5.  autolink: http://www.example.com/
                case 6.  email address: test@example.com
                case 7.  reference: example
                case 8.  collapsed link: example
                case 9.  shortcut link: example
                case 10. inline without URL: example
                case 11. reference: foo
                case 12. reference: foo
                case 13. collapsed link: foo
                case 14. shortcut link: foo
                case 15. inline without URL: foo
                case 16. just escaped text: \[foo]
                case 17. inline link: Foo

                [^example]: https://www.example.com/
            "#]],
        );
    }

    #[test]
    fn test_hover_macro_generated_struct_fn_doc_comment() {
        cov_mark::check!(hover_macro_generated_struct_fn_doc_comment);

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

fn foo() { let bar = Bar; bar.fo$0o(); }
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
        cov_mark::check!(hover_macro_generated_struct_fn_doc_attr);

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

fn foo() { let bar = Bar; bar.fo$0o(); }
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
            r#"trait foo$0() {}"#,
            expect![[r#"
                [
                    Implementation(
                        FilePosition {
                            file_id: FileId(
                                0,
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
            r"struct foo$0() {}",
            expect![[r#"
                [
                    Implementation(
                        FilePosition {
                            file_id: FileId(
                                0,
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
            r#"union foo$0() {}"#,
            expect![[r#"
                [
                    Implementation(
                        FilePosition {
                            file_id: FileId(
                                0,
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
            r"enum foo$0() { A, B }",
            expect![[r#"
                [
                    Implementation(
                        FilePosition {
                            file_id: FileId(
                                0,
                            ),
                            offset: 5,
                        },
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_self_has_impl_action() {
        check_actions(
            r#"struct foo where Self$0:;"#,
            expect![[r#"
                [
                    Implementation(
                        FilePosition {
                            file_id: FileId(
                                0,
                            ),
                            offset: 7,
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
fn foo_$0test() {}
"#,
            expect![[r#"
                [
                    Runnable(
                        Runnable {
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..24,
                                focus_range: 11..19,
                                name: "foo_test",
                                kind: Function,
                            },
                            kind: Test {
                                test_id: Path(
                                    "foo_test",
                                ),
                                attr: TestAttr {
                                    ignore: false,
                                },
                            },
                            cfg: None,
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
mod tests$0 {
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
                                    0,
                                ),
                                full_range: 0..46,
                                focus_range: 4..9,
                                name: "tests",
                                kind: Module,
                            },
                            kind: TestMod {
                                path: "tests",
                            },
                            cfg: None,
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

fn main() { let s$0t = S{ f1:0 }; }
            "#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..19,
                                    focus_range: 7..8,
                                    name: "S",
                                    kind: Struct,
                                    description: "struct S",
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

fn main() { let s$0t = S{ f1:Arg(0) }; }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 17..37,
                                    focus_range: 24..25,
                                    name: "S",
                                    kind: Struct,
                                    description: "struct S",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::Arg",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..16,
                                    focus_range: 7..10,
                                    name: "Arg",
                                    kind: Struct,
                                    description: "struct Arg",
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

fn main() { let s$0t = S{ f1: S{ f1: Arg(0) } }; }
            "#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 17..37,
                                    focus_range: 24..25,
                                    name: "S",
                                    kind: Struct,
                                    description: "struct S",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::Arg",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..16,
                                    focus_range: 7..10,
                                    name: "Arg",
                                    kind: Struct,
                                    description: "struct Arg",
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

fn main() { let s$0t = (A(1), B(2), M::C(3) ); }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::A",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..14,
                                    focus_range: 7..8,
                                    name: "A",
                                    kind: Struct,
                                    description: "struct A",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::B",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 15..29,
                                    focus_range: 22..23,
                                    name: "B",
                                    kind: Struct,
                                    description: "struct B",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::M::C",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 42..60,
                                    focus_range: 53..54,
                                    name: "C",
                                    kind: Struct,
                                    description: "pub struct C",
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

fn main() { let s$0t = foo(); }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..12,
                                    focus_range: 6..9,
                                    name: "Foo",
                                    kind: Trait,
                                    description: "trait Foo",
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

fn main() { let s$0t = foo(); }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..15,
                                    focus_range: 6..9,
                                    name: "Foo",
                                    kind: Trait,
                                    description: "trait Foo",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 16..25,
                                    focus_range: 23..24,
                                    name: "S",
                                    kind: Struct,
                                    description: "struct S",
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

fn main() { let s$0t = foo(); }
            "#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..12,
                                    focus_range: 6..9,
                                    name: "Foo",
                                    kind: Trait,
                                    description: "trait Foo",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::Bar",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 13..25,
                                    focus_range: 19..22,
                                    name: "Bar",
                                    kind: Trait,
                                    description: "trait Bar",
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

fn main() { let s$0t = foo(); }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..15,
                                    focus_range: 6..9,
                                    name: "Foo",
                                    kind: Trait,
                                    description: "trait Foo",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::Bar",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 16..31,
                                    focus_range: 22..25,
                                    name: "Bar",
                                    kind: Trait,
                                    description: "trait Bar",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S1",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 32..44,
                                    focus_range: 39..41,
                                    name: "S1",
                                    kind: Struct,
                                    description: "struct S1",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S2",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 45..57,
                                    focus_range: 52..54,
                                    name: "S2",
                                    kind: Struct,
                                    description: "struct S2",
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
fn foo(ar$0g: &impl Foo) {}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..12,
                                    focus_range: 6..9,
                                    name: "Foo",
                                    kind: Trait,
                                    description: "trait Foo",
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

fn foo(ar$0g: &impl Foo + Bar<S>) {}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..12,
                                    focus_range: 6..9,
                                    name: "Foo",
                                    kind: Trait,
                                    description: "trait Foo",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::Bar",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 13..28,
                                    focus_range: 19..22,
                                    name: "Bar",
                                    kind: Trait,
                                    description: "trait Bar",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 29..39,
                                    focus_range: 36..37,
                                    name: "S",
                                    kind: Struct,
                                    description: "struct S",
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
    let fo$0o = async { S };
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
                                        0,
                                    ),
                                    full_range: 101..163,
                                    focus_range: 140..146,
                                    name: "Future",
                                    kind: Trait,
                                    description: "pub trait Future",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..9,
                                    focus_range: 7..8,
                                    name: "S",
                                    kind: Struct,
                                    description: "struct S",
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
fn foo(ar$0g: &impl Foo<S>) {}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..15,
                                    focus_range: 6..9,
                                    name: "Foo",
                                    kind: Trait,
                                    description: "trait Foo",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 16..27,
                                    focus_range: 23..24,
                                    name: "S",
                                    kind: Struct,
                                    description: "struct S",
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

fn main() { let s$0t = foo(); }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::B",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 42..55,
                                    focus_range: 49..50,
                                    name: "B",
                                    kind: Struct,
                                    description: "struct B",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..12,
                                    focus_range: 6..9,
                                    name: "Foo",
                                    kind: Trait,
                                    description: "trait Foo",
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
fn foo(ar$0g: &dyn Foo) {}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..12,
                                    focus_range: 6..9,
                                    name: "Foo",
                                    kind: Trait,
                                    description: "trait Foo",
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
fn foo(ar$0g: &dyn Foo<S>) {}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..15,
                                    focus_range: 6..9,
                                    name: "Foo",
                                    kind: Trait,
                                    description: "trait Foo",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 16..27,
                                    focus_range: 23..24,
                                    name: "S",
                                    kind: Struct,
                                    description: "struct S",
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

fn foo(a$0rg: &impl ImplTrait<B<dyn DynTrait<B<S>>>>) {}
            "#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::ImplTrait",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..21,
                                    focus_range: 6..15,
                                    name: "ImplTrait",
                                    kind: Trait,
                                    description: "trait ImplTrait",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::B",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 43..57,
                                    focus_range: 50..51,
                                    name: "B",
                                    kind: Struct,
                                    description: "struct B",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::DynTrait",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 22..42,
                                    focus_range: 28..36,
                                    name: "DynTrait",
                                    kind: Trait,
                                    description: "trait DynTrait",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::S",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 58..69,
                                    focus_range: 65..66,
                                    name: "S",
                                    kind: Struct,
                                    description: "struct S",
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

fn main() { let s$0t = test().get(); }
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..62,
                                    focus_range: 6..9,
                                    name: "Foo",
                                    kind: Trait,
                                    description: "trait Foo",
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_const_param_has_goto_type_action() {
        check_actions(
            r#"
struct Bar;
struct Foo<const BAR: Bar>;

impl<const BAR: Bar> Foo<BAR$0> {}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Bar",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..11,
                                    focus_range: 7..10,
                                    name: "Bar",
                                    kind: Struct,
                                    description: "struct Bar",
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_type_param_has_goto_type_action() {
        check_actions(
            r#"
trait Foo {}

fn foo<T: Foo>(t: T$0){}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..12,
                                    focus_range: 6..9,
                                    name: "Foo",
                                    kind: Trait,
                                    description: "trait Foo",
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn test_hover_self_has_go_to_type() {
        check_actions(
            r#"
struct Foo;
impl Foo {
    fn foo(&self$0) {}
}
"#,
            expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..11,
                                    focus_range: 7..10,
                                    name: "Foo",
                                    kind: Struct,
                                    description: "struct Foo",
                                },
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn hover_displays_normalized_crate_names() {
        check(
            r#"
//- /lib.rs crate:name-with-dashes
pub mod wrapper {
    pub struct Thing { x: u32 }

    impl Thing {
        pub fn new() -> Thing { Thing { x: 0 } }
    }
}

//- /main.rs crate:main deps:name-with-dashes
fn main() { let foo_test = name_with_dashes::wrapper::Thing::new$0(); }
"#,
            expect![[r#"
            *new*

            ```rust
            name_with_dashes::wrapper::Thing
            ```

            ```rust
            pub fn new() -> Thing
            ```
            "#]],
        )
    }

    #[test]
    fn hover_field_pat_shorthand_ref_match_ergonomics() {
        check(
            r#"
struct S {
    f: i32,
}

fn main() {
    let s = S { f: 0 };
    let S { f$0 } = &s;
}
"#,
            expect![[r#"
                *f*

                ```rust
                f: &i32
                ```
            "#]],
        );
    }

    #[test]
    fn hover_self_param_shows_type() {
        check(
            r#"
struct Foo {}
impl Foo {
    fn bar(&sel$0f) {}
}
"#,
            expect![[r#"
                *self*

                ```rust
                self: &Foo
                ```
            "#]],
        );
    }

    #[test]
    fn hover_self_param_shows_type_for_arbitrary_self_type() {
        check(
            r#"
struct Arc<T>(T);
struct Foo {}
impl Foo {
    fn bar(sel$0f: Arc<Foo>) {}
}
"#,
            expect![[r#"
                *self*

                ```rust
                self: Arc<Foo>
                ```
            "#]],
        );
    }

    #[test]
    fn hover_doc_outer_inner() {
        check(
            r#"
/// Be quick;
mod Foo$0 {
    //! time is mana

    /// This comment belongs to the function
    fn foo() {}
}
"#,
            expect![[r#"
                *Foo*

                ```rust
                test
                ```

                ```rust
                mod Foo
                ```

                ---

                Be quick;
                time is mana
            "#]],
        );
    }

    #[test]
    fn hover_doc_outer_inner_attribue() {
        check(
            r#"
#[doc = "Be quick;"]
mod Foo$0 {
    #![doc = "time is mana"]

    #[doc = "This comment belongs to the function"]
    fn foo() {}
}
"#,
            expect![[r#"
                *Foo*

                ```rust
                test
                ```

                ```rust
                mod Foo
                ```

                ---

                Be quick;
                time is mana
            "#]],
        );
    }

    #[test]
    fn hover_comments_dont_highlight_parent() {
        check_hover_no_result(
            r#"
fn no_hover() {
    // no$0hover
}
"#,
        );
    }

    #[test]
    fn hover_label() {
        check(
            r#"
fn foo() {
    'label$0: loop {}
}
"#,
            expect![[r#"
            *'label*

            ```rust
            'label
            ```
            "#]],
        );
    }

    #[test]
    fn hover_lifetime() {
        check(
            r#"fn foo<'lifetime>(_: &'lifetime$0 ()) {}"#,
            expect![[r#"
            *'lifetime*

            ```rust
            'lifetime
            ```
            "#]],
        );
    }

    #[test]
    fn hover_type_param() {
        check(
            r#"
struct Foo<T>(T);
trait Copy {}
trait Clone {}
trait Sized {}
impl<T: Copy + Clone> Foo<T$0> where T: Sized {}
"#,
            expect![[r#"
                *T*

                ```rust
                T: Copy + Clone + Sized
                ```
            "#]],
        );
        check(
            r#"
struct Foo<T>(T);
impl<T> Foo<T$0> {}
"#,
            expect![[r#"
                *T*

                ```rust
                T
                ```
                "#]],
        );
        // lifetimes bounds arent being tracked yet
        check(
            r#"
struct Foo<T>(T);
impl<T: 'static> Foo<T$0> {}
"#,
            expect![[r#"
                *T*

                ```rust
                T
                ```
                "#]],
        );
    }

    #[test]
    fn hover_const_param() {
        check(
            r#"
struct Foo<const LEN: usize>;
impl<const LEN: usize> Foo<LEN$0> {}
"#,
            expect![[r#"
                *LEN*

                ```rust
                const LEN: usize
                ```
            "#]],
        );
    }

    #[test]
    fn hover_const_pat() {
        check(
            r#"
/// This is a doc
const FOO: usize = 3;
fn foo() {
    match 5 {
        FOO$0 => (),
        _ => ()
    }
}
"#,
            expect![[r#"
                *FOO*

                ```rust
                test
                ```

                ```rust
                const FOO: usize
                ```

                ---

                This is a doc
            "#]],
        );
    }

    #[test]
    fn hover_mod_def() {
        check(
            r#"
//- /main.rs
mod foo$0;
//- /foo.rs
//! For the horde!
"#,
            expect![[r#"
                *foo*
                For the horde!
            "#]],
        );
    }

    #[test]
    fn hover_self_in_use() {
        check(
            r#"
//! This should not appear
mod foo {
    /// But this should appear
    pub mod bar {}
}
use foo::bar::{self$0};
"#,
            expect![[r#"
                *self*

                ```rust
                test::foo
                ```

                ```rust
                pub mod bar
                ```

                ---

                But this should appear
            "#]],
        )
    }

    #[test]
    fn hover_keyword() {
        let ra_fixture = r#"//- /main.rs crate:main deps:std
fn f() { retur$0n; }"#;
        let fixture = format!("{}\n{}", ra_fixture, FamousDefs::FIXTURE);
        check(
            &fixture,
            expect![[r#"
                *return*

                ```rust
                return
                ```

                ---

                Docs for return_keyword
            "#]],
        );
    }

    #[test]
    fn hover_builtin() {
        let ra_fixture = r#"//- /main.rs crate:main deps:std
cosnt _: &str$0 = ""; }"#;
        let fixture = format!("{}\n{}", ra_fixture, FamousDefs::FIXTURE);
        check(
            &fixture,
            expect![[r#"
                *str*

                ```rust
                str
                ```

                ---

                Docs for prim_str
            "#]],
        );
    }

    #[test]
    fn hover_macro_expanded_function() {
        check(
            r#"
struct S<'a, T>(&'a T);
trait Clone {}
macro_rules! foo {
    () => {
        fn bar<'t, T: Clone + 't>(s: &mut S<'t, T>, t: u32) -> *mut u32 where
            't: 't + 't,
            for<'a> T: Clone + 'a
        { 0 as _ }
    };
}

foo!();

fn main() {
    bar$0;
}
"#,
            expect![[r#"
                *bar*

                ```rust
                test
                ```

                ```rust
                fn bar<'t, T: Clone + 't>(s: &mut S<'t, T>, t: u32) -> *mut u32
                where
                    't: 't + 't,
                    for<'a> T: Clone + 'a,
                ```
            "#]],
        )
    }
}
