use either::Either;
use hir::{AssocItem, Enum, HasVisibility, Module, ModuleDef, Name, PathResolution, ScopeDef};
use ide_db::{
    defs::{Definition, NameRefClass},
    search::SearchScope,
    source_change::SourceChangeBuilder,
};
use stdx::never;
use syntax::{
    AstNode, Direction, SyntaxNode, SyntaxToken, T,
    ast::{self, Use, UseTree, VisibilityKind, make},
    ted,
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
};

// Assist: expand_glob_import
//
// Expands glob imports.
//
// ```
// mod foo {
//     pub struct Bar;
//     pub struct Baz;
// }
//
// use foo::*$0;
//
// fn qux(bar: Bar, baz: Baz) {}
// ```
// ->
// ```
// mod foo {
//     pub struct Bar;
//     pub struct Baz;
// }
//
// use foo::{Bar, Baz};
//
// fn qux(bar: Bar, baz: Baz) {}
// ```
pub(crate) fn expand_glob_import(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let star = ctx.find_token_syntax_at_offset(T![*])?;
    let use_tree = star.parent().and_then(ast::UseTree::cast)?;
    let use_item = star.parent_ancestors().find_map(ast::Use::cast)?;
    let (parent, mod_path) = find_parent_and_path(&star)?;
    let target_module = match ctx.sema.resolve_path(&mod_path)? {
        PathResolution::Def(ModuleDef::Module(it)) => Expandable::Module(it),
        PathResolution::Def(ModuleDef::Adt(hir::Adt::Enum(e))) => Expandable::Enum(e),
        _ => return None,
    };

    let current_scope = ctx.sema.scope(&star.parent()?)?;
    let current_module = current_scope.module();

    if !is_visible_from(ctx, &target_module, current_module) {
        return None;
    }

    let target = parent.either(|n| n.syntax().clone(), |n| n.syntax().clone());
    acc.add(
        AssistId::refactor_rewrite("expand_glob_import"),
        "Expand glob import",
        target.text_range(),
        |builder| {
            build_expanded_import(
                ctx,
                builder,
                use_tree,
                use_item,
                target_module,
                current_module,
                false,
            )
        },
    )
}

// Assist: expand_glob_reexport
//
// Expands non-private glob imports.
//
// ```
// mod foo {
//     pub struct Bar;
//     pub struct Baz;
// }
//
// pub use foo::*$0;
// ```
// ->
// ```
// mod foo {
//     pub struct Bar;
//     pub struct Baz;
// }
//
// pub use foo::{Bar, Baz};
// ```
pub(crate) fn expand_glob_reexport(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let star = ctx.find_token_syntax_at_offset(T![*])?;
    let use_tree = star.parent().and_then(ast::UseTree::cast)?;
    let use_item = star.parent_ancestors().find_map(ast::Use::cast)?;
    let (parent, mod_path) = find_parent_and_path(&star)?;
    let target_module = match ctx.sema.resolve_path(&mod_path)? {
        PathResolution::Def(ModuleDef::Module(it)) => Expandable::Module(it),
        PathResolution::Def(ModuleDef::Adt(hir::Adt::Enum(e))) => Expandable::Enum(e),
        _ => return None,
    };

    let current_scope = ctx.sema.scope(&star.parent()?)?;
    let current_module = current_scope.module();

    if let VisibilityKind::PubSelf = get_export_visibility_kind(&use_item) {
        return None;
    }
    if !is_visible_from(ctx, &target_module, current_module) {
        return None;
    }

    let target = parent.either(|n| n.syntax().clone(), |n| n.syntax().clone());
    acc.add(
        AssistId::refactor_rewrite("expand_glob_reexport"),
        "Expand glob reexport",
        target.text_range(),
        |builder| {
            build_expanded_import(
                ctx,
                builder,
                use_tree,
                use_item,
                target_module,
                current_module,
                true,
            )
        },
    )
}

fn build_expanded_import(
    ctx: &AssistContext<'_>,
    builder: &mut SourceChangeBuilder,
    use_tree: UseTree,
    use_item: Use,
    target_module: Expandable,
    current_module: Module,
    reexport_public_items: bool,
) {
    let (must_be_pub, visible_from) = if !reexport_public_items {
        (false, current_module)
    } else {
        match get_export_visibility_kind(&use_item) {
            VisibilityKind::Pub => (true, current_module.krate().root_module()),
            VisibilityKind::PubCrate => (false, current_module.krate().root_module()),
            _ => (false, current_module),
        }
    };

    let refs_in_target = find_refs_in_mod(ctx, target_module, visible_from, must_be_pub);
    let imported_defs = find_imported_defs(ctx, use_item);

    let filtered_defs =
        if reexport_public_items { refs_in_target } else { refs_in_target.used_refs(ctx) };

    let use_tree = builder.make_mut(use_tree);

    let names_to_import = find_names_to_import(filtered_defs, imported_defs);
    let expanded = make::use_tree_list(names_to_import.iter().map(|n| {
        let path = make::ext::ident_path(
            &n.display(ctx.db(), current_module.krate().edition(ctx.db())).to_string(),
        );
        make::use_tree(path, None, None, false)
    }))
    .clone_for_update();

    match use_tree.star_token() {
        Some(star) => {
            let needs_braces = use_tree.path().is_some() && names_to_import.len() != 1;
            if needs_braces {
                ted::replace(star, expanded.syntax())
            } else {
                let without_braces = expanded
                    .syntax()
                    .children_with_tokens()
                    .filter(|child| !matches!(child.kind(), T!['{'] | T!['}']))
                    .collect();
                ted::replace_with_many(star, without_braces)
            }
        }
        None => never!(),
    }
}

fn get_export_visibility_kind(use_item: &Use) -> VisibilityKind {
    use syntax::ast::HasVisibility as _;
    match use_item.visibility() {
        Some(vis) => match vis.kind() {
            VisibilityKind::PubCrate => VisibilityKind::PubCrate,
            VisibilityKind::Pub => VisibilityKind::Pub,
            VisibilityKind::PubSelf => VisibilityKind::PubSelf,
            // We don't handle pub(in ...) and pub(super) yet
            VisibilityKind::In(_) => VisibilityKind::PubSelf,
            VisibilityKind::PubSuper => VisibilityKind::PubSelf,
        },
        None => VisibilityKind::PubSelf,
    }
}

enum Expandable {
    Module(Module),
    Enum(Enum),
}

fn find_parent_and_path(
    star: &SyntaxToken,
) -> Option<(Either<ast::UseTree, ast::UseTreeList>, ast::Path)> {
    return star.parent_ancestors().find_map(|n| {
        find_use_tree_list(n.clone())
            .map(|(u, p)| (Either::Right(u), p))
            .or_else(|| find_use_tree(n).map(|(u, p)| (Either::Left(u), p)))
    });

    fn find_use_tree_list(n: SyntaxNode) -> Option<(ast::UseTreeList, ast::Path)> {
        let use_tree_list = ast::UseTreeList::cast(n)?;
        let path = use_tree_list.parent_use_tree().path()?;
        Some((use_tree_list, path))
    }

    fn find_use_tree(n: SyntaxNode) -> Option<(ast::UseTree, ast::Path)> {
        let use_tree = ast::UseTree::cast(n)?;
        let path = use_tree.path()?;
        Some((use_tree, path))
    }
}

fn def_is_referenced_in(def: Definition, ctx: &AssistContext<'_>) -> bool {
    let search_scope = SearchScope::single_file(ctx.file_id());
    def.usages(&ctx.sema).in_scope(&search_scope).at_least_one()
}

#[derive(Debug, Clone)]
struct Ref {
    // could be alias
    visible_name: Name,
    def: Definition,
    is_pub: bool,
}

impl Ref {
    fn from_scope_def(ctx: &AssistContext<'_>, name: Name, scope_def: ScopeDef) -> Option<Self> {
        match scope_def {
            ScopeDef::ModuleDef(def) => Some(Ref {
                visible_name: name,
                def: Definition::from(def),
                is_pub: matches!(def.visibility(ctx.db()), hir::Visibility::Public),
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct Refs(Vec<Ref>);

impl Refs {
    fn used_refs(&self, ctx: &AssistContext<'_>) -> Refs {
        Refs(
            self.0
                .clone()
                .into_iter()
                .filter(|r| {
                    if let Definition::Trait(tr) = r.def
                        && tr.items(ctx.db()).into_iter().any(|ai| {
                            if let AssocItem::Function(f) = ai {
                                def_is_referenced_in(Definition::Function(f), ctx)
                            } else {
                                false
                            }
                        })
                    {
                        return true;
                    }

                    def_is_referenced_in(r.def, ctx)
                })
                .collect(),
        )
    }

    fn filter_out_by_defs(&self, defs: Vec<Definition>) -> Refs {
        Refs(self.0.clone().into_iter().filter(|r| !defs.contains(&r.def)).collect())
    }
}

fn find_refs_in_mod(
    ctx: &AssistContext<'_>,
    expandable: Expandable,
    visible_from: Module,
    must_be_pub: bool,
) -> Refs {
    match expandable {
        Expandable::Module(module) => {
            let module_scope = module.scope(ctx.db(), Some(visible_from));
            let refs = module_scope
                .into_iter()
                .filter_map(|(n, d)| Ref::from_scope_def(ctx, n, d))
                .filter(|r| !must_be_pub || r.is_pub)
                .collect();
            Refs(refs)
        }
        Expandable::Enum(enm) => Refs(
            enm.variants(ctx.db())
                .into_iter()
                .map(|v| Ref {
                    visible_name: v.name(ctx.db()),
                    def: Definition::Variant(v),
                    is_pub: true,
                })
                .collect(),
        ),
    }
}

fn is_visible_from(ctx: &AssistContext<'_>, expandable: &Expandable, from: Module) -> bool {
    fn is_mod_visible_from(ctx: &AssistContext<'_>, module: Module, from: Module) -> bool {
        match module.parent(ctx.db()) {
            Some(parent) => {
                module.visibility(ctx.db()).is_visible_from(ctx.db(), from.into())
                    && is_mod_visible_from(ctx, parent, from)
            }
            None => true,
        }
    }

    match expandable {
        Expandable::Module(module) => match module.parent(ctx.db()) {
            Some(parent) => {
                module.visibility(ctx.db()).is_visible_from(ctx.db(), from.into())
                    && is_mod_visible_from(ctx, parent, from)
            }
            None => true,
        },
        Expandable::Enum(enm) => {
            let module = enm.module(ctx.db());
            enm.visibility(ctx.db()).is_visible_from(ctx.db(), from.into())
                && is_mod_visible_from(ctx, module, from)
        }
    }
}

// looks for name refs in parent use block's siblings
//
// mod bar {
//     mod qux {
//         struct Qux;
//     }
//
//     pub use qux::Qux;
// }
//
// ↓ ---------------
// use foo::*$0;
// use baz::Baz;
// ↑ ---------------
fn find_imported_defs(ctx: &AssistContext<'_>, use_item: Use) -> Vec<Definition> {
    [Direction::Prev, Direction::Next]
        .into_iter()
        .flat_map(|dir| {
            use_item.syntax().siblings(dir.to_owned()).filter(|n| ast::Use::can_cast(n.kind()))
        })
        .flat_map(|n| n.descendants().filter_map(ast::NameRef::cast))
        .filter_map(|r| match NameRefClass::classify(&ctx.sema, &r)? {
            NameRefClass::Definition(
                def @ (Definition::Macro(_)
                | Definition::Module(_)
                | Definition::Function(_)
                | Definition::Adt(_)
                | Definition::Variant(_)
                | Definition::Const(_)
                | Definition::Static(_)
                | Definition::Trait(_)
                | Definition::TypeAlias(_)),
                _,
            ) => Some(def),
            _ => None,
        })
        .collect()
}

fn find_names_to_import(refs_in_target: Refs, imported_defs: Vec<Definition>) -> Vec<Name> {
    let final_refs = refs_in_target.filter_out_by_defs(imported_defs);
    final_refs.0.iter().map(|r| r.visible_name.clone()).collect()
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn expanding_glob_import() {
        check_assist(
            expand_glob_import,
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    pub struct Qux;

    pub fn f() {}
}

use foo::*$0;

fn qux(bar: Bar, baz: Baz) {
    f();
}
",
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    pub struct Qux;

    pub fn f() {}
}

use foo::{Bar, Baz, f};

fn qux(bar: Bar, baz: Baz) {
    f();
}
",
        )
    }

    #[test]
    fn expanding_glob_import_unused() {
        check_assist(
            expand_glob_import,
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    pub struct Qux;

    pub fn f() {}
}

use foo::*$0;

fn qux() {}
",
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    pub struct Qux;

    pub fn f() {}
}

use foo::{};

fn qux() {}
",
        )
    }

    #[test]
    fn expanding_glob_import_with_existing_explicit_names() {
        check_assist(
            expand_glob_import,
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    pub struct Qux;

    pub fn f() {}
}

use foo::{*$0, f};

fn qux(bar: Bar, baz: Baz) {
    f();
}
",
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    pub struct Qux;

    pub fn f() {}
}

use foo::{Bar, Baz, f};

fn qux(bar: Bar, baz: Baz) {
    f();
}
",
        )
    }

    #[test]
    fn expanding_glob_import_with_existing_uses_in_same_module() {
        check_assist(
            expand_glob_import,
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    pub struct Qux;

    pub fn f() {}
}

use foo::Bar;
use foo::{*$0, f};

fn qux(bar: Bar, baz: Baz) {
    f();
}
",
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    pub struct Qux;

    pub fn f() {}
}

use foo::Bar;
use foo::{Baz, f};

fn qux(bar: Bar, baz: Baz) {
    f();
}
",
        )
    }

    #[test]
    fn expanding_nested_glob_import() {
        check_assist(
            expand_glob_import,
            r"
mod foo {
    pub mod bar {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;

        pub fn f() {}
    }

    pub mod baz {
        pub fn g() {}
    }
}

use foo::{bar::{*$0, f}, baz::*};

fn qux(bar: Bar, baz: Baz) {
    f();
    g();
}
",
            r"
mod foo {
    pub mod bar {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;

        pub fn f() {}
    }

    pub mod baz {
        pub fn g() {}
    }
}

use foo::{bar::{Bar, Baz, f}, baz::*};

fn qux(bar: Bar, baz: Baz) {
    f();
    g();
}
",
        );

        check_assist(
            expand_glob_import,
            r"
mod foo {
    pub mod bar {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;

        pub fn f() {}
    }

    pub mod baz {
        pub fn g() {}
    }
}

use foo::{bar::{Bar, Baz, f}, baz::*$0};

fn qux(bar: Bar, baz: Baz) {
    f();
    g();
}
",
            r"
mod foo {
    pub mod bar {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;

        pub fn f() {}
    }

    pub mod baz {
        pub fn g() {}
    }
}

use foo::{bar::{Bar, Baz, f}, baz::g};

fn qux(bar: Bar, baz: Baz) {
    f();
    g();
}
",
        );

        check_assist(
            expand_glob_import,
            r"
mod foo {
    pub mod bar {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;

        pub fn f() {}
    }

    pub mod baz {
        pub fn g() {}

        pub mod qux {
            pub fn h() {}
            pub fn m() {}

            pub mod q {
                pub fn j() {}
            }
        }
    }
}

use foo::{
    bar::{*, f},
    baz::{g, qux::*$0}
};

fn qux(bar: Bar, baz: Baz) {
    f();
    g();
    h();
    q::j();
}
",
            r"
mod foo {
    pub mod bar {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;

        pub fn f() {}
    }

    pub mod baz {
        pub fn g() {}

        pub mod qux {
            pub fn h() {}
            pub fn m() {}

            pub mod q {
                pub fn j() {}
            }
        }
    }
}

use foo::{
    bar::{*, f},
    baz::{g, qux::{h, q}}
};

fn qux(bar: Bar, baz: Baz) {
    f();
    g();
    h();
    q::j();
}
",
        );

        check_assist(
            expand_glob_import,
            r"
mod foo {
    pub mod bar {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;

        pub fn f() {}
    }

    pub mod baz {
        pub fn g() {}

        pub mod qux {
            pub fn h() {}
            pub fn m() {}

            pub mod q {
                pub fn j() {}
            }
        }
    }
}

use foo::{
    bar::{*, f},
    baz::{g, qux::{h, q::*$0}}
};

fn qux(bar: Bar, baz: Baz) {
    f();
    g();
    h();
    j();
}
",
            r"
mod foo {
    pub mod bar {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;

        pub fn f() {}
    }

    pub mod baz {
        pub fn g() {}

        pub mod qux {
            pub fn h() {}
            pub fn m() {}

            pub mod q {
                pub fn j() {}
            }
        }
    }
}

use foo::{
    bar::{*, f},
    baz::{g, qux::{h, q::j}}
};

fn qux(bar: Bar, baz: Baz) {
    f();
    g();
    h();
    j();
}
",
        );

        check_assist(
            expand_glob_import,
            r"
mod foo {
    pub mod bar {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;

        pub fn f() {}
    }

    pub mod baz {
        pub fn g() {}

        pub mod qux {
            pub fn h() {}
            pub fn m() {}

            pub mod q {
                pub fn j() {}
            }
        }
    }
}

use foo::{
    bar::{*, f},
    baz::{g, qux::{q::j, *$0}}
};

fn qux(bar: Bar, baz: Baz) {
    f();
    g();
    h();
    j();
}
",
            r"
mod foo {
    pub mod bar {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;

        pub fn f() {}
    }

    pub mod baz {
        pub fn g() {}

        pub mod qux {
            pub fn h() {}
            pub fn m() {}

            pub mod q {
                pub fn j() {}
            }
        }
    }
}

use foo::{
    bar::{*, f},
    baz::{g, qux::{q::j, h}}
};

fn qux(bar: Bar, baz: Baz) {
    f();
    g();
    h();
    j();
}
",
        );
    }

    #[test]
    fn expanding_glob_import_with_macro_defs() {
        check_assist(
            expand_glob_import,
            r#"
//- /lib.rs crate:foo
#[macro_export]
macro_rules! bar {
    () => ()
}

pub fn baz() {}

//- /main.rs crate:main deps:foo
use foo::*$0;

fn main() {
    bar!();
    baz();
}
"#,
            r#"
use foo::{bar, baz};

fn main() {
    bar!();
    baz();
}
"#,
        );
    }

    #[test]
    fn expanding_glob_import_with_trait_method_uses() {
        check_assist(
            expand_glob_import,
            r"
//- /lib.rs crate:foo
pub trait Tr {
    fn method(&self) {}
}
impl Tr for () {}

//- /main.rs crate:main deps:foo
use foo::*$0;

fn main() {
    ().method();
}
",
            r"
use foo::Tr;

fn main() {
    ().method();
}
",
        );

        check_assist(
            expand_glob_import,
            r"
//- /lib.rs crate:foo
pub trait Tr {
    fn method(&self) {}
}
impl Tr for () {}

pub trait Tr2 {
    fn method2(&self) {}
}
impl Tr2 for () {}

//- /main.rs crate:main deps:foo
use foo::*$0;

fn main() {
    ().method();
}
",
            r"
use foo::Tr;

fn main() {
    ().method();
}
",
        );
    }

    #[test]
    fn expanding_is_not_applicable_if_target_module_is_not_accessible_from_current_scope() {
        check_assist_not_applicable(
            expand_glob_import,
            r"
mod foo {
    mod bar {
        pub struct Bar;
    }
}

use foo::bar::*$0;

fn baz(bar: Bar) {}
",
        );

        check_assist_not_applicable(
            expand_glob_import,
            r"
mod foo {
    mod bar {
        pub mod baz {
            pub struct Baz;
        }
    }
}

use foo::bar::baz::*$0;

fn qux(baz: Baz) {}
",
        );
    }

    #[test]
    fn expanding_is_not_applicable_if_cursor_is_not_in_star_token() {
        check_assist_not_applicable(
            expand_glob_import,
            r"
    mod foo {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;
    }

    use foo::Bar$0;

    fn qux(bar: Bar, baz: Baz) {}
    ",
        )
    }

    #[test]
    fn expanding_glob_import_single_nested_glob_only() {
        check_assist(
            expand_glob_import,
            r"
mod foo {
    pub struct Bar;
}

use foo::{*$0};

struct Baz {
    bar: Bar
}
",
            r"
mod foo {
    pub struct Bar;
}

use foo::{Bar};

struct Baz {
    bar: Bar
}
",
        );
    }

    #[test]
    fn test_support_for_enums() {
        check_assist(
            expand_glob_import,
            r#"
mod foo {
    pub enum Foo {
        Bar,
        Baz,
    }
}

use foo::Foo;
use foo::Foo::*$0;

struct Strukt {
    bar: Foo,
}

fn main() {
    let s: Strukt = Strukt { bar: Bar };
}"#,
            r#"
mod foo {
    pub enum Foo {
        Bar,
        Baz,
    }
}

use foo::Foo;
use foo::Foo::Bar;

struct Strukt {
    bar: Foo,
}

fn main() {
    let s: Strukt = Strukt { bar: Bar };
}"#,
        )
    }

    #[test]
    fn test_expanding_multiple_variants_at_once() {
        check_assist(
            expand_glob_import,
            r#"
mod foo {
    pub enum Foo {
        Bar,
        Baz,
    }
}

mod abc {
    use super::foo;
    use super::foo::Foo::*$0;

    struct Strukt {
        baz: foo::Foo,
        bar: foo::Foo,
    }

    fn trying_calling() {
        let s: Strukt = Strukt { bar: Bar , baz : Baz };
    }

}"#,
            r#"
mod foo {
    pub enum Foo {
        Bar,
        Baz,
    }
}

mod abc {
    use super::foo;
    use super::foo::Foo::{Bar, Baz};

    struct Strukt {
        baz: foo::Foo,
        bar: foo::Foo,
    }

    fn trying_calling() {
        let s: Strukt = Strukt { bar: Bar , baz : Baz };
    }

}"#,
        )
    }

    #[test]
    fn expanding_glob_reexport() {
        check_assist(
            expand_glob_reexport,
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    struct Qux;

    pub fn f() {}

    pub(crate) fn g() {}
    pub(self) fn h() {}
}

pub use foo::*$0;
",
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    struct Qux;

    pub fn f() {}

    pub(crate) fn g() {}
    pub(self) fn h() {}
}

pub use foo::{Bar, Baz, f};
",
        )
    }

    #[test]
    fn expanding_recursive_glob_reexport() {
        check_assist(
            expand_glob_reexport,
            r"
mod foo {
    pub use bar::*;
    mod bar {
        pub struct Bar;
        pub struct Baz;
    }
}

pub use foo::*$0;
",
            r"
mod foo {
    pub use bar::*;
    mod bar {
        pub struct Bar;
        pub struct Baz;
    }
}

pub use foo::{Bar, Baz};
",
        )
    }

    #[test]
    fn expanding_reexport_is_not_applicable_for_private_import() {
        check_assist_not_applicable(
            expand_glob_reexport,
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
}

use foo::*$0;
",
        );
    }
}
