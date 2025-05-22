use std::collections::hash_map::Entry;

use hir::{
    FileRange, InFile, InRealFile, Module, ModuleDef, ModuleSource, PathResolution,
    PathResolutionPerNs,
};
use ide_db::text_edit::TextRange;
use ide_db::{
    FxHashMap, RootDatabase,
    defs::Definition,
    search::{FileReference, ReferenceCategory, SearchScope},
};
use syntax::{
    AstNode,
    ast::{self, Rename},
};

use crate::{AssistContext, AssistId, Assists};

// Assist: remove_unused_imports
//
// Removes any use statements in the current selection that are unused.
//
// ```
// struct X();
// mod foo {
//     use super::X$0;
// }
// ```
// ->
// ```
// struct X();
// mod foo {
// }
// ```
pub(crate) fn remove_unused_imports(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    // First, grab the uses that intersect with the current selection.
    let selected_el = match ctx.covering_element() {
        syntax::NodeOrToken::Node(n) => n,
        syntax::NodeOrToken::Token(t) => t.parent()?,
    };

    // This applies to all uses that are selected, or are ancestors of our selection.
    let uses_up = selected_el.ancestors().skip(1).filter_map(ast::Use::cast);
    let uses_down = selected_el
        .descendants()
        .filter(|x| x.text_range().intersect(ctx.selection_trimmed()).is_some())
        .filter_map(ast::Use::cast);
    let uses = uses_up.chain(uses_down).collect::<Vec<_>>();

    // Maps use nodes to the scope that we should search through to find
    let mut search_scopes = FxHashMap::<Module, Vec<SearchScope>>::default();

    // iterator over all unused use trees
    let mut unused = uses
        .into_iter()
        .flat_map(|u| u.syntax().descendants().filter_map(ast::UseTree::cast))
        .filter(|u| u.use_tree_list().is_none())
        .filter_map(|u| {
            // Find any uses trees that are unused

            let use_module = ctx.sema.scope(u.syntax()).map(|s| s.module())?;
            let scope = match search_scopes.entry(use_module) {
                Entry::Occupied(o) => o.into_mut(),
                Entry::Vacant(v) => v.insert(module_search_scope(ctx.db(), use_module)),
            };

            // Gets the path associated with this use tree. If there isn't one, then ignore this use tree.
            let path = if let Some(path) = u.path() {
                path
            } else if u.star_token().is_some() {
                // This case maps to the situation where the * token is braced.
                // In this case, the parent use tree's path is the one we should use to resolve the glob.
                match u.syntax().ancestors().skip(1).find_map(ast::UseTree::cast) {
                    Some(parent_u) if parent_u.path().is_some() => parent_u.path()?,
                    _ => return None,
                }
            } else {
                return None;
            };

            // Get the actual definition associated with this use item.
            let res = match ctx.sema.resolve_path_per_ns(&path) {
                Some(x) if x.any().is_some() => x,
                Some(_) | None => {
                    return None;
                }
            };

            if u.star_token().is_some() {
                // Check if any of the children of this module are used
                let def_mod = match res.type_ns {
                    Some(PathResolution::Def(ModuleDef::Module(module))) => module,
                    _ => return None,
                };

                if !def_mod
                    .scope(ctx.db(), Some(use_module))
                    .iter()
                    .filter_map(|(_, x)| match x {
                        hir::ScopeDef::ModuleDef(d) => Some(Definition::from(*d)),
                        _ => None,
                    })
                    .any(|d| used_once_in_scope(ctx, d, u.rename(), scope))
                {
                    Some(u)
                } else {
                    None
                }
            } else {
                is_path_per_ns_unused_in_scope(ctx, &u, scope, &res).then_some(u)
            }
        })
        .peekable();

    // Peek so we terminate early if an unused use is found. Only do the rest of the work if the user selects the assist.
    if unused.peek().is_some() {
        acc.add(
            AssistId::quick_fix("remove_unused_imports"),
            "Remove all the unused imports",
            selected_el.text_range(),
            |builder| {
                let unused: Vec<ast::UseTree> = unused.map(|x| builder.make_mut(x)).collect();
                for node in unused {
                    node.remove_recursive();
                }
            },
        )
    } else {
        None
    }
}

fn is_path_per_ns_unused_in_scope(
    ctx: &AssistContext<'_>,
    u: &ast::UseTree,
    scope: &mut Vec<SearchScope>,
    path: &PathResolutionPerNs,
) -> bool {
    if let Some(PathResolution::Def(ModuleDef::Trait(ref t))) = path.type_ns {
        if is_trait_unused_in_scope(ctx, u, scope, t) {
            let path = [path.value_ns, path.macro_ns];
            is_path_unused_in_scope(ctx, u, scope, &path)
        } else {
            false
        }
    } else {
        let path = [path.type_ns, path.value_ns, path.macro_ns];
        is_path_unused_in_scope(ctx, u, scope, &path)
    }
}

fn is_path_unused_in_scope(
    ctx: &AssistContext<'_>,
    u: &ast::UseTree,
    scope: &mut Vec<SearchScope>,
    path: &[Option<PathResolution>],
) -> bool {
    !path
        .iter()
        .filter_map(|path| *path)
        .filter_map(|res| match res {
            PathResolution::Def(d) => Some(Definition::from(d)),
            _ => None,
        })
        .any(|def| used_once_in_scope(ctx, def, u.rename(), scope))
}

fn is_trait_unused_in_scope(
    ctx: &AssistContext<'_>,
    u: &ast::UseTree,
    scope: &mut Vec<SearchScope>,
    t: &hir::Trait,
) -> bool {
    !std::iter::once((Definition::Trait(*t), u.rename()))
        .chain(t.items(ctx.db()).into_iter().map(|item| (item.into(), None)))
        .any(|(d, rename)| used_once_in_scope(ctx, d, rename, scope))
}

fn used_once_in_scope(
    ctx: &AssistContext<'_>,
    def: Definition,
    rename: Option<Rename>,
    scopes: &Vec<SearchScope>,
) -> bool {
    let mut found = false;

    for scope in scopes {
        let mut search_non_import = |_, r: FileReference| {
            // The import itself is a use; we must skip that.
            if !r.category.contains(ReferenceCategory::IMPORT) {
                found = true;
                true
            } else {
                false
            }
        };
        def.usages(&ctx.sema)
            .in_scope(scope)
            .with_rename(rename.as_ref())
            .search(&mut search_non_import);
        if found {
            break;
        }
    }

    found
}

/// Build a search scope spanning the given module but none of its submodules.
fn module_search_scope(db: &RootDatabase, module: hir::Module) -> Vec<SearchScope> {
    let (file_id, range) = {
        let InFile { file_id, value } = module.definition_source(db);
        if let Some(InRealFile { file_id, value: call_source }) = file_id.original_call_node(db) {
            (file_id, Some(call_source.text_range()))
        } else {
            (
                file_id.original_file(db),
                match value {
                    ModuleSource::SourceFile(_) => None,
                    ModuleSource::Module(it) => Some(it.syntax().text_range()),
                    ModuleSource::BlockExpr(it) => Some(it.syntax().text_range()),
                },
            )
        }
    };

    fn split_at_subrange(first: TextRange, second: TextRange) -> (TextRange, Option<TextRange>) {
        let intersect = first.intersect(second);
        if let Some(intersect) = intersect {
            let start_range = TextRange::new(first.start(), intersect.start());

            if intersect.end() < first.end() {
                (start_range, Some(TextRange::new(intersect.end(), first.end())))
            } else {
                (start_range, None)
            }
        } else {
            (first, None)
        }
    }

    let mut scopes = Vec::new();
    if let Some(range) = range {
        let mut ranges = vec![range];

        for child in module.children(db) {
            let rng = match child.definition_source(db).value {
                ModuleSource::SourceFile(_) => continue,
                ModuleSource::Module(it) => it.syntax().text_range(),
                ModuleSource::BlockExpr(_) => continue,
            };
            let mut new_ranges = Vec::new();
            for old_range in ranges.iter_mut() {
                let split = split_at_subrange(*old_range, rng);
                *old_range = split.0;
                new_ranges.extend(split.1);
            }

            ranges.append(&mut new_ranges);
        }

        for range in ranges {
            scopes.push(SearchScope::file_range(FileRange { file_id, range }));
        }
    } else {
        scopes.push(SearchScope::single_file(file_id));
    }

    scopes
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn remove_unused() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
struct Y();
mod z {
    $0use super::X;
    use super::Y;$0
}
"#,
            r#"
struct X();
struct Y();
mod z {
}
"#,
        );
    }

    #[test]
    fn remove_unused_is_precise() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
mod z {
$0use super::X;$0

fn w() {
    struct X();
    let x = X();
}
}
"#,
            r#"
struct X();
mod z {

fn w() {
    struct X();
    let x = X();
}
}
"#,
        );
    }

    #[test]
    fn trait_name_use_is_use() {
        check_assist_not_applicable(
            remove_unused_imports,
            r#"
struct X();
trait Y {
    fn f();
}

impl Y for X {
    fn f() {}
}
mod z {
$0use super::X;
use super::Y;$0

fn w() {
    X::f();
}
}
"#,
        );
    }

    #[test]
    fn trait_item_use_is_use() {
        check_assist_not_applicable(
            remove_unused_imports,
            r#"
struct X();
trait Y {
    fn f(self);
}

impl Y for X {
    fn f(self) {}
}
mod z {
$0use super::X;
use super::Y;$0

fn w() {
    let x = X();
    x.f();
}
}
"#,
        );
    }

    #[test]
    fn renamed_trait_item_use_is_use() {
        check_assist_not_applicable(
            remove_unused_imports,
            r#"
struct X();
trait Y {
    fn f(self);
}

impl Y for X {
    fn f(self) {}
}
mod z {
$0use super::X;
use super::Y as Z;$0

fn w() {
    let x = X();
    x.f();
}
}
"#,
        );
    }

    #[test]
    fn renamed_underscore_trait_item_use_is_use() {
        check_assist_not_applicable(
            remove_unused_imports,
            r#"
struct X();
trait Y {
    fn f(self);
}

impl Y for X {
    fn f(self) {}
}
mod z {
$0use super::X;
use super::Y as _;$0

fn w() {
    let x = X();
    x.f();
}
}
"#,
        );
    }

    #[test]
    fn dont_remove_used() {
        check_assist_not_applicable(
            remove_unused_imports,
            r#"
struct X();
struct Y();
mod z {
$0use super::X;
use super::Y;$0

fn w() {
    let x = X();
    let y = Y();
}
}
"#,
        );
    }

    #[test]
    fn remove_unused_in_braces() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
struct Y();
mod z {
    $0use super::{X, Y};$0

    fn w() {
        let x = X();
    }
}
"#,
            r#"
struct X();
struct Y();
mod z {
    use super::X;

    fn w() {
        let x = X();
    }
}
"#,
        );
    }

    #[test]
    fn remove_unused_under_cursor() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
mod z {
    use super::X$0;
}
"#,
            r#"
struct X();
mod z {
}
"#,
        );
    }

    #[test]
    fn remove_multi_use_block() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
$0mod y {
    use super::X;
}
mod z {
    use super::X;
}$0
"#,
            r#"
struct X();
mod y {
}
mod z {
}
"#,
        );
    }

    #[test]
    fn remove_nested() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
mod y {
    struct Y();
    mod z {
        use crate::{X, y::Y}$0;
        fn f() {
            let x = X();
        }
    }
}
"#,
            r#"
struct X();
mod y {
    struct Y();
    mod z {
        use crate::X;
        fn f() {
            let x = X();
        }
    }
}
"#,
        );
    }

    #[test]
    fn remove_nested_first_item() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
mod y {
    struct Y();
    mod z {
        use crate::{X, y::Y}$0;
        fn f() {
            let y = Y();
        }
    }
}
"#,
            r#"
struct X();
mod y {
    struct Y();
    mod z {
        use crate::y::Y;
        fn f() {
            let y = Y();
        }
    }
}
"#,
        );
    }

    #[test]
    fn remove_unused_auto_remove_brace_nested() {
        check_assist(
            remove_unused_imports,
            r#"
mod a {
    pub struct A();
}
mod b {
    struct F();
    mod c {
        $0use {{super::{{
            {d::{{{{{{{S, U}}}}}}}},
            {{{{e::{H, L, {{{R}}}}}}}},
            F, super::a::A
        }}}};$0
        fn f() {
            let f = F();
            let l = L();
            let a = A();
            let s = S();
            let h = H();
        }
    }

    mod d {
        pub struct S();
        pub struct U();
    }

    mod e {
        pub struct H();
        pub struct L();
        pub struct R();
    }
}
"#,
            r#"
mod a {
    pub struct A();
}
mod b {
    struct F();
    mod c {
        use super::{
            d::S,
            e::{H, L},
            F, super::a::A
        };
        fn f() {
            let f = F();
            let l = L();
            let a = A();
            let s = S();
            let h = H();
        }
    }

    mod d {
        pub struct S();
        pub struct U();
    }

    mod e {
        pub struct H();
        pub struct L();
        pub struct R();
    }
}
"#,
        );
    }

    #[test]
    fn remove_comma_after_auto_remove_brace() {
        check_assist(
            remove_unused_imports,
            r#"
mod m {
    pub mod x {
        pub struct A;
        pub struct B;
    }
    pub mod y {
        pub struct C;
    }
}

$0use m::{
    x::{A, B},
    y::C,
};$0

fn main() {
    B;
}
"#,
            r#"
mod m {
    pub mod x {
        pub struct A;
        pub struct B;
    }
    pub mod y {
        pub struct C;
    }
}

use m::
    x::B
;

fn main() {
    B;
}
"#,
        );
        check_assist(
            remove_unused_imports,
            r#"
mod m {
    pub mod x {
        pub struct A;
        pub struct B;
    }
    pub mod y {
        pub struct C;
        pub struct D;
    }
    pub mod z {
        pub struct E;
        pub struct F;
    }
}

$0use m::{
    x::{A, B},
    y::{C, D,},
    z::{E, F},
};$0

fn main() {
    B;
    C;
    F;
}
"#,
            r#"
mod m {
    pub mod x {
        pub struct A;
        pub struct B;
    }
    pub mod y {
        pub struct C;
        pub struct D;
    }
    pub mod z {
        pub struct E;
        pub struct F;
    }
}

use m::{
    x::B,
    y::C,
    z::F,
};

fn main() {
    B;
    C;
    F;
}
"#,
        );
    }

    #[test]
    fn remove_nested_all_unused() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
mod y {
    struct Y();
    mod z {
        use crate::{X, y::Y}$0;
    }
}
"#,
            r#"
struct X();
mod y {
    struct Y();
    mod z {
    }
}
"#,
        );
    }

    #[test]
    fn remove_unused_glob() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
struct Y();
mod z {
    use super::*$0;
}
"#,
            r#"
struct X();
struct Y();
mod z {
}
"#,
        );
    }

    #[test]
    fn remove_unused_braced_glob() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
struct Y();
mod z {
    use super::{*}$0;
}
"#,
            r#"
struct X();
struct Y();
mod z {
}
"#,
        );
    }

    #[test]
    fn remove_unused_fixes_nested_self() {
        check_assist(
            remove_unused_imports,
            r#"
mod inner {
    pub struct X();
    pub struct Y();
}

mod z {
    use super::inner::{self, X}$0;

    fn f() {
        let y = inner::Y();
    }
}
"#,
            r#"mod inner {
    pub struct X();
    pub struct Y();
}

mod z {
    use super::inner::{self};

    fn f() {
        let y = inner::Y();
    }
}
"#,
        );
    }

    #[test]
    fn dont_remove_used_glob() {
        check_assist_not_applicable(
            remove_unused_imports,
            r#"
struct X();
struct Y();
mod z {
    use super::*$0;

    fn f() {
        let x = X();
    }
}
"#,
        );
    }

    #[test]
    fn only_remove_from_selection() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
struct Y();
mod z {
    $0use super::X;$0
    use super::Y;
}
mod w {
    use super::Y;
}
"#,
            r#"
struct X();
struct Y();
mod z {
    use super::Y;
}
mod w {
    use super::Y;
}
"#,
        );
    }

    #[test]
    fn test_several_files() {
        check_assist(
            remove_unused_imports,
            r#"
//- /foo.rs
pub struct X();
pub struct Y();

//- /main.rs
$0use foo::X;
use foo::Y;
$0
mod foo;
mod z {
    use crate::foo::X;
}
"#,
            r#"

mod foo;
mod z {
    use crate::foo::X;
}
"#,
        );
    }

    #[test]
    fn use_in_submodule_doesnt_count() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
mod z {
    use super::X$0;

    mod w {
        use crate::X;

        fn f() {
            let x = X();
        }
    }
}
"#,
            r#"
struct X();
mod z {

    mod w {
        use crate::X;

        fn f() {
            let x = X();
        }
    }
}
"#,
        );
    }

    #[test]
    fn use_in_submodule_file_doesnt_count() {
        check_assist(
            remove_unused_imports,
            r#"
//- /z/foo.rs
use crate::X;
fn f() {
    let x = X();
}

//- /main.rs
pub struct X();

mod z {
    use crate::X$0;
    mod foo;
}
"#,
            r#"
pub struct X();

mod z {
    mod foo;
}
"#,
        );
    }

    #[test]
    fn use_as_alias() {
        check_assist_not_applicable(
            remove_unused_imports,
            r#"
mod foo {
    pub struct Foo {}
}

use foo::Foo as Bar$0;

fn test(_: Bar) {}
"#,
        );

        check_assist(
            remove_unused_imports,
            r#"
mod foo {
    pub struct Foo {}
    pub struct Bar {}
    pub struct Qux {}
    pub trait Quux {
        fn quxx(&self) {}
    }
    impl<T> Quxx for T {}
}

use foo::{Foo as Bar, Bar as Baz, Qux as _, Quxx as _}$0;

fn test(_: Bar) {
    let a = ();
    a.quxx();
}
"#,
            r#"
mod foo {
    pub struct Foo {}
    pub struct Bar {}
    pub struct Qux {}
    pub trait Quux {
        fn quxx(&self) {}
    }
    impl<T> Quxx for T {}
}

use foo::{Foo as Bar, Quxx as _};

fn test(_: Bar) {
    let a = ();
    a.quxx();
}
"#,
        );
    }

    #[test]
    fn test_unused_macro() {
        check_assist(
            remove_unused_imports,
            r#"
//- /foo.rs crate:foo
#[macro_export]
macro_rules! m { () => {} }

//- /main.rs crate:main deps:foo
use foo::m;$0
fn main() {}
"#,
            r#"
fn main() {}
"#,
        );

        check_assist_not_applicable(
            remove_unused_imports,
            r#"
//- /foo.rs crate:foo
#[macro_export]
macro_rules! m { () => {} }

//- /main.rs crate:main deps:foo
use foo::m;$0
fn main() {
    m!();
}
"#,
        );

        check_assist_not_applicable(
            remove_unused_imports,
            r#"
//- /foo.rs crate:foo
#[macro_export]
macro_rules! m { () => {} }

//- /bar.rs crate:bar deps:foo
pub use foo::m;
fn m() {}


//- /main.rs crate:main deps:bar
use bar::m;$0
fn main() {
    m!();
}
"#,
        );
    }

    #[test]
    fn test_conflict_derive_macro() {
        check_assist_not_applicable(
            remove_unused_imports,
            r#"
//- proc_macros: derive_identity
//- minicore: derive
//- /bar.rs crate:bar
pub use proc_macros::DeriveIdentity;
pub trait DeriveIdentity {}

//- /main.rs crate:main deps:bar
$0use bar::DeriveIdentity;$0
#[derive(DeriveIdentity)]
struct S;
"#,
        );

        check_assist_not_applicable(
            remove_unused_imports,
            r#"
//- proc_macros: derive_identity
//- minicore: derive
//- /bar.rs crate:bar
pub use proc_macros::DeriveIdentity;
pub fn DeriveIdentity() {}

//- /main.rs crate:main deps:bar
$0use bar::DeriveIdentity;$0
#[derive(DeriveIdentity)]
struct S;
"#,
        );

        check_assist_not_applicable(
            remove_unused_imports,
            r#"
//- proc_macros: derive_identity
//- minicore: derive
//- /bar.rs crate:bar
pub use proc_macros::DeriveIdentity;
pub fn DeriveIdentity() {}

//- /main.rs crate:main deps:bar
$0use bar::DeriveIdentity;$0
fn main() {
    DeriveIdentity();
}
"#,
        );
    }
}
