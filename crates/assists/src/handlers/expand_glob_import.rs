use either::Either;
use std::iter::successors;

use hir::{AssocItem, MacroDef, ModuleDef, Name, PathResolution, ScopeDef, SemanticsScope};
use ide_db::{
    defs::{classify_name_ref, Definition, NameRefClass},
    RootDatabase,
};
use syntax::{algo, ast, AstNode, SourceFile, SyntaxNode, SyntaxToken, T};

use crate::{
    assist_context::{AssistBuilder, AssistContext, Assists},
    AssistId, AssistKind,
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
// use foo::*<|>;
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
// use foo::{Baz, Bar};
//
// fn qux(bar: Bar, baz: Baz) {}
// ```
pub(crate) fn expand_glob_import(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let star = ctx.find_token_at_offset(T![*])?;
    let (parent, mod_path) = find_parent_and_path(&star)?;
    let module = match ctx.sema.resolve_path(&mod_path)? {
        PathResolution::Def(ModuleDef::Module(it)) => it,
        _ => return None,
    };

    let source_file = ctx.source_file();
    let scope = ctx.sema.scope_at_offset(source_file.syntax(), ctx.offset());

    let defs_in_mod = find_defs_in_mod(ctx, scope, module)?;
    let names_to_import = find_names_to_import(ctx, source_file, defs_in_mod);

    let target = parent.clone().either(|n| n.syntax().clone(), |n| n.syntax().clone());
    acc.add(
        AssistId("expand_glob_import", AssistKind::RefactorRewrite),
        "Expand glob import",
        target.text_range(),
        |builder| {
            replace_ast(builder, parent, mod_path, names_to_import);
        },
    )
}

fn find_parent_and_path(
    star: &SyntaxToken,
) -> Option<(Either<ast::UseTree, ast::UseTreeList>, ast::Path)> {
    return star.ancestors().find_map(|n| {
        find_use_tree_list(n.clone())
            .and_then(|(u, p)| Some((Either::Right(u), p)))
            .or_else(|| find_use_tree(n).and_then(|(u, p)| Some((Either::Left(u), p))))
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

#[derive(PartialEq)]
enum Def {
    ModuleDef(ModuleDef),
    MacroDef(MacroDef),
}

impl Def {
    fn name(&self, db: &RootDatabase) -> Option<Name> {
        match self {
            Def::ModuleDef(def) => def.name(db),
            Def::MacroDef(def) => def.name(db),
        }
    }
}

fn find_defs_in_mod(
    ctx: &AssistContext,
    from: SemanticsScope<'_>,
    module: hir::Module,
) -> Option<Vec<Def>> {
    let module_scope = module.scope(ctx.db(), from.module());

    let mut defs = vec![];
    for (_, def) in module_scope {
        match def {
            ScopeDef::ModuleDef(def) => defs.push(Def::ModuleDef(def)),
            ScopeDef::MacroDef(def) => defs.push(Def::MacroDef(def)),
            _ => continue,
        }
    }

    Some(defs)
}

fn find_names_to_import(
    ctx: &AssistContext,
    source_file: &SourceFile,
    defs_in_mod: Vec<Def>,
) -> Vec<Name> {
    let (name_refs_in_use_item, name_refs_in_source) = source_file
        .syntax()
        .descendants()
        .filter_map(|n| {
            let name_ref = ast::NameRef::cast(n.clone())?;
            let name_ref_class = classify_name_ref(&ctx.sema, &name_ref)?;
            let is_in_use_item =
                successors(n.parent(), |n| n.parent()).find_map(ast::Use::cast).is_some();
            Some((name_ref_class, is_in_use_item))
        })
        .partition::<Vec<_>, _>(|&(_, is_in_use_item)| is_in_use_item);

    let name_refs_to_import: Vec<NameRefClass> = name_refs_in_source
        .into_iter()
        .filter_map(|(r, _)| {
            if name_refs_in_use_item.contains(&(r.clone(), true)) {
                // already imported
                return None;
            }
            Some(r)
        })
        .collect();

    let defs_in_source_file = name_refs_to_import
        .into_iter()
        .filter_map(|rc| match rc {
            NameRefClass::Definition(Definition::ModuleDef(def)) => Some(Def::ModuleDef(def)),
            NameRefClass::Definition(Definition::Macro(def)) => Some(Def::MacroDef(def)),
            _ => None,
        })
        .collect::<Vec<Def>>();

    defs_in_mod
        .iter()
        .filter(|def| {
            if let Def::ModuleDef(ModuleDef::Trait(tr)) = def {
                for item in tr.items(ctx.db()) {
                    if let AssocItem::Function(f) = item {
                        if defs_in_source_file.contains(&Def::ModuleDef(ModuleDef::Function(f))) {
                            return true;
                        }
                    }
                }
            }

            defs_in_source_file.contains(def)
        })
        .filter_map(|d| d.name(ctx.db()))
        .collect()
}

fn replace_ast(
    builder: &mut AssistBuilder,
    parent: Either<ast::UseTree, ast::UseTreeList>,
    path: ast::Path,
    names_to_import: Vec<Name>,
) {
    let existing_use_trees = match parent.clone() {
        Either::Left(_) => vec![],
        Either::Right(u) => u.use_trees().filter(|n| 
            // filter out star
            n.star_token().is_none()
        ).collect(),
    };

    let new_use_trees: Vec<ast::UseTree> = names_to_import
        .iter()
        .map(|n| ast::make::use_tree(ast::make::path_from_text(&n.to_string()), None, None, false))
        .collect();

    let use_trees = [&existing_use_trees[..], &new_use_trees[..]].concat();

    match use_trees.as_slice() {
        [name] => {
            if let Some(end_path) = name.path() {
                let replacement = ast::make::use_tree(
                    ast::make::path_from_text(&format!("{}::{}", path, end_path)),
                    None,
                    None,
                    false,
                );

                algo::diff(
                    &parent.either(|n| n.syntax().clone(), |n| n.syntax().clone()),
                    replacement.syntax(),
                )
                .into_text_edit(builder.text_edit_builder());
            }
        }
        names => {
            let replacement = match parent {
                Either::Left(_) => ast::make::use_tree(
                    path,
                    Some(ast::make::use_tree_list(names.to_owned())),
                    None,
                    false,
                )
                .syntax()
                .clone(),
                Either::Right(_) => ast::make::use_tree_list(names.to_owned()).syntax().clone(),
            };

            algo::diff(
                &parent.either(|n| n.syntax().clone(), |n| n.syntax().clone()),
                &replacement,
            )
            .into_text_edit(builder.text_edit_builder());
        }
    };
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

use foo::*<|>;

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

use foo::{Baz, Bar, f};

fn qux(bar: Bar, baz: Baz) {
    f();
}
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

use foo::{*<|>, f};

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

use foo::{f, Baz, Bar};

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
use foo::{*<|>, f};

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
use foo::{f, Baz};

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

use foo::{bar::{*<|>, f}, baz::*};

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

use foo::{bar::{f, Baz, Bar}, baz::*};

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

use foo::{bar::{Bar, Baz, f}, baz::*<|>};

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
    baz::{g, qux::*<|>}
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
    baz::{g, qux::{q, h}}
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
    baz::{g, qux::{h, q::*<|>}}
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
    baz::{g, qux::{q::j, *<|>}}
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
            r"
//- /lib.rs crate:foo
#[macro_export]
macro_rules! bar {
    () => ()
}

pub fn baz() {}

//- /main.rs crate:main deps:foo
use foo::*<|>;

fn main() {
    bar!();
    baz();
}
",
            r"
use foo::{bar, baz};

fn main() {
    bar!();
    baz();
}
",
        )
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
use foo::*<|>;

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
        )
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

    use foo::Bar<|>;

    fn qux(bar: Bar, baz: Baz) {}
    ",
        )
    }
}
