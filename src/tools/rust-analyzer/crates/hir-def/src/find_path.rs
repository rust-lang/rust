//! An algorithm to find a path to refer to a certain item.

use std::{cell::Cell, cmp::Ordering, iter};

use hir_expand::{
    name::{known, AsName, Name},
    Lookup,
};
use rustc_hash::FxHashSet;

use crate::{
    db::DefDatabase,
    item_scope::ItemInNs,
    nameres::DefMap,
    path::{ModPath, PathKind},
    visibility::{Visibility, VisibilityExplicitness},
    ImportPathConfig, ModuleDefId, ModuleId,
};

/// Find a path that can be used to refer to a certain item. This can depend on
/// *from where* you're referring to the item, hence the `from` parameter.
pub fn find_path(
    db: &dyn DefDatabase,
    item: ItemInNs,
    from: ModuleId,
    mut prefix_kind: PrefixKind,
    ignore_local_imports: bool,
    mut cfg: ImportPathConfig,
) -> Option<ModPath> {
    let _p = tracing::info_span!("find_path").entered();

    // - if the item is a builtin, it's in scope
    if let ItemInNs::Types(ModuleDefId::BuiltinType(builtin)) = item {
        return Some(ModPath::from_segments(PathKind::Plain, iter::once(builtin.as_name())));
    }

    // within block modules, forcing a `self` or `crate` prefix will not allow using inner items, so
    // default to plain paths.
    if item.module(db).is_some_and(ModuleId::is_within_block) {
        prefix_kind = PrefixKind::Plain;
    }
    cfg.prefer_no_std = cfg.prefer_no_std || db.crate_supports_no_std(from.krate());

    find_path_inner(
        &FindPathCtx {
            db,
            prefix: prefix_kind,
            cfg,
            ignore_local_imports,
            from,
            from_def_map: &from.def_map(db),
            fuel: Cell::new(FIND_PATH_FUEL),
        },
        item,
        MAX_PATH_LEN,
    )
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Stability {
    Unstable,
    Stable,
}
use Stability::*;

fn zip_stability(a: Stability, b: Stability) -> Stability {
    match (a, b) {
        (Stable, Stable) => Stable,
        _ => Unstable,
    }
}

const MAX_PATH_LEN: usize = 15;
const FIND_PATH_FUEL: usize = 10000;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrefixKind {
    /// Causes paths to always start with either `self`, `super`, `crate` or a crate-name.
    /// This is the same as plain, just that paths will start with `self` prepended if the path
    /// starts with an identifier that is not a crate.
    BySelf,
    /// Causes paths to not use a self, super or crate prefix.
    Plain,
    /// Causes paths to start with `crate` where applicable, effectively forcing paths to be absolute.
    ByCrate,
}

impl PrefixKind {
    #[inline]
    fn path_kind(self) -> PathKind {
        match self {
            PrefixKind::BySelf => PathKind::SELF,
            PrefixKind::Plain => PathKind::Plain,
            PrefixKind::ByCrate => PathKind::Crate,
        }
    }
}

struct FindPathCtx<'db> {
    db: &'db dyn DefDatabase,
    prefix: PrefixKind,
    cfg: ImportPathConfig,
    ignore_local_imports: bool,
    from: ModuleId,
    from_def_map: &'db DefMap,
    fuel: Cell<usize>,
}

/// Attempts to find a path to refer to the given `item` visible from the `from` ModuleId
fn find_path_inner(ctx: &FindPathCtx<'_>, item: ItemInNs, max_len: usize) -> Option<ModPath> {
    // - if the item is a module, jump straight to module search
    if let ItemInNs::Types(ModuleDefId::ModuleId(module_id)) = item {
        let mut visited_modules = FxHashSet::default();
        return find_path_for_module(ctx, &mut visited_modules, module_id, max_len)
            .map(|(item, _)| item);
    }

    let may_be_in_scope = match ctx.prefix {
        PrefixKind::Plain | PrefixKind::BySelf => true,
        PrefixKind::ByCrate => ctx.from.is_crate_root(),
    };
    if may_be_in_scope {
        // - if the item is already in scope, return the name under which it is
        let scope_name =
            find_in_scope(ctx.db, ctx.from_def_map, ctx.from, item, ctx.ignore_local_imports);
        if let Some(scope_name) = scope_name {
            return Some(ModPath::from_segments(ctx.prefix.path_kind(), iter::once(scope_name)));
        }
    }

    // - if the item is in the prelude, return the name from there
    if let Some(value) = find_in_prelude(ctx.db, ctx.from_def_map, item, ctx.from) {
        return Some(value);
    }

    if let Some(ModuleDefId::EnumVariantId(variant)) = item.as_module_def_id() {
        // - if the item is an enum variant, refer to it via the enum
        if let Some(mut path) =
            find_path_inner(ctx, ItemInNs::Types(variant.lookup(ctx.db).parent.into()), max_len)
        {
            path.push_segment(ctx.db.enum_variant_data(variant).name.clone());
            return Some(path);
        }
        // If this doesn't work, it seems we have no way of referring to the
        // enum; that's very weird, but there might still be a reexport of the
        // variant somewhere
    }

    let mut visited_modules = FxHashSet::default();

    calculate_best_path(ctx, &mut visited_modules, item, max_len).map(|(item, _)| item)
}

#[tracing::instrument(skip_all)]
fn find_path_for_module(
    ctx: &FindPathCtx<'_>,
    visited_modules: &mut FxHashSet<ModuleId>,
    module_id: ModuleId,
    max_len: usize,
) -> Option<(ModPath, Stability)> {
    if let Some(crate_root) = module_id.as_crate_root() {
        if crate_root == ctx.from.derive_crate_root() {
            // - if the item is the crate root, return `crate`
            return Some((ModPath::from_segments(PathKind::Crate, None), Stable));
        }
        // - otherwise if the item is the crate root of a dependency crate, return the name from the extern prelude

        let root_def_map = ctx.from.derive_crate_root().def_map(ctx.db);
        // rev here so we prefer looking at renamed extern decls first
        for (name, (def_id, _extern_crate)) in root_def_map.extern_prelude().rev() {
            if crate_root != def_id {
                continue;
            }
            let name_already_occupied_in_type_ns = ctx
                .from_def_map
                .with_ancestor_maps(ctx.db, ctx.from.local_id, &mut |def_map, local_id| {
                    def_map[local_id]
                        .scope
                        .type_(name)
                        .filter(|&(id, _)| id != ModuleDefId::ModuleId(def_id.into()))
                })
                .is_some();
            let kind = if name_already_occupied_in_type_ns {
                cov_mark::hit!(ambiguous_crate_start);
                PathKind::Abs
            } else {
                PathKind::Plain
            };
            return Some((ModPath::from_segments(kind, iter::once(name.clone())), Stable));
        }
    }

    let may_be_in_scope = match ctx.prefix {
        PrefixKind::Plain | PrefixKind::BySelf => true,
        PrefixKind::ByCrate => ctx.from.is_crate_root(),
    };
    if may_be_in_scope {
        let scope_name = find_in_scope(
            ctx.db,
            ctx.from_def_map,
            ctx.from,
            ItemInNs::Types(module_id.into()),
            ctx.ignore_local_imports,
        );
        if let Some(scope_name) = scope_name {
            // - if the item is already in scope, return the name under which it is
            return Some((
                ModPath::from_segments(ctx.prefix.path_kind(), iter::once(scope_name)),
                Stable,
            ));
        }
    }

    // - if the module can be referenced as self, super or crate, do that
    if let Some(mod_path) = is_kw_kind_relative_to_from(ctx.from_def_map, module_id, ctx.from) {
        if ctx.prefix != PrefixKind::ByCrate || mod_path.kind == PathKind::Crate {
            return Some((mod_path, Stable));
        }
    }

    // - if the module is in the prelude, return it by that path
    if let Some(mod_path) =
        find_in_prelude(ctx.db, ctx.from_def_map, ItemInNs::Types(module_id.into()), ctx.from)
    {
        return Some((mod_path, Stable));
    }
    calculate_best_path(ctx, visited_modules, ItemInNs::Types(module_id.into()), max_len)
}

fn find_in_scope(
    db: &dyn DefDatabase,
    def_map: &DefMap,
    from: ModuleId,
    item: ItemInNs,
    ignore_local_imports: bool,
) -> Option<Name> {
    // FIXME: We could have multiple applicable names here, but we currently only return the first
    def_map.with_ancestor_maps(db, from.local_id, &mut |def_map, local_id| {
        def_map[local_id].scope.names_of(item, |name, _, declared| {
            (declared || !ignore_local_imports).then(|| name.clone())
        })
    })
}

/// Returns single-segment path (i.e. without any prefix) if `item` is found in prelude and its
/// name doesn't clash in current scope.
fn find_in_prelude(
    db: &dyn DefDatabase,
    local_def_map: &DefMap,
    item: ItemInNs,
    from: ModuleId,
) -> Option<ModPath> {
    let (prelude_module, _) = local_def_map.prelude()?;
    let prelude_def_map = prelude_module.def_map(db);
    let prelude_scope = &prelude_def_map[prelude_module.local_id].scope;
    let (name, vis, _declared) = prelude_scope.name_of(item)?;
    if !vis.is_visible_from(db, from) {
        return None;
    }

    // Check if the name is in current scope and it points to the same def.
    let found_and_same_def =
        local_def_map.with_ancestor_maps(db, from.local_id, &mut |def_map, local_id| {
            let per_ns = def_map[local_id].scope.get(name);
            let same_def = match item {
                ItemInNs::Types(it) => per_ns.take_types()? == it,
                ItemInNs::Values(it) => per_ns.take_values()? == it,
                ItemInNs::Macros(it) => per_ns.take_macros()? == it,
            };
            Some(same_def)
        });

    if found_and_same_def.unwrap_or(true) {
        Some(ModPath::from_segments(PathKind::Plain, iter::once(name.clone())))
    } else {
        None
    }
}

fn is_kw_kind_relative_to_from(
    def_map: &DefMap,
    item: ModuleId,
    from: ModuleId,
) -> Option<ModPath> {
    if item.krate != from.krate || item.is_within_block() || from.is_within_block() {
        return None;
    }
    let item = item.local_id;
    let from = from.local_id;
    if item == from {
        // - if the item is the module we're in, use `self`
        Some(ModPath::from_segments(PathKind::SELF, None))
    } else if let Some(parent_id) = def_map[from].parent {
        if item == parent_id {
            // - if the item is the parent module, use `super` (this is not used recursively, since `super::super` is ugly)
            Some(ModPath::from_segments(
                if parent_id == DefMap::ROOT { PathKind::Crate } else { PathKind::Super(1) },
                None,
            ))
        } else {
            None
        }
    } else {
        None
    }
}

#[tracing::instrument(skip_all)]
fn calculate_best_path(
    ctx: &FindPathCtx<'_>,
    visited_modules: &mut FxHashSet<ModuleId>,
    item: ItemInNs,
    max_len: usize,
) -> Option<(ModPath, Stability)> {
    if max_len <= 1 {
        // recursive base case, we can't find a path prefix of length 0, one segment is occupied by
        // the item's name itself.
        return None;
    }
    let fuel = ctx.fuel.get();
    if fuel == 0 {
        // we ran out of fuel, so we stop searching here
        tracing::warn!(
            "ran out of fuel while searching for a path for item {item:?} of krate {:?} from krate {:?}",
            item.krate(ctx.db),
            ctx.from.krate()
        );
        return None;
    }
    ctx.fuel.set(fuel - 1);

    let mut best_path = None;
    let mut best_path_len = max_len;
    let mut process = |mut path: (ModPath, Stability), name, best_path_len: &mut _| {
        path.0.push_segment(name);
        let new_path = match best_path.take() {
            Some(best_path) => select_best_path(best_path, path, ctx.cfg),
            None => path,
        };
        if new_path.1 == Stable {
            *best_path_len = new_path.0.len();
        }
        match &mut best_path {
            Some((old_path, old_stability)) => {
                *old_path = new_path.0;
                *old_stability = zip_stability(*old_stability, new_path.1);
            }
            None => best_path = Some(new_path),
        }
    };
    let db = ctx.db;
    if item.krate(db) == Some(ctx.from.krate) {
        // Item was defined in the same crate that wants to import it. It cannot be found in any
        // dependency in this case.
        // FIXME: cache the `find_local_import_locations` output?
        find_local_import_locations(db, item, ctx.from, ctx.from_def_map, |name, module_id| {
            if !visited_modules.insert(module_id) {
                return;
            }
            // we are looking for paths of length up to best_path_len, any longer will make it be
            // less optimal. The -1 is due to us pushing name onto it afterwards.
            if let Some(path) =
                find_path_for_module(ctx, visited_modules, module_id, best_path_len - 1)
            {
                process(path, name.clone(), &mut best_path_len);
            }
        })
    } else {
        // Item was defined in some upstream crate. This means that it must be exported from one,
        // too (unless we can't name it at all). It could *also* be (re)exported by the same crate
        // that wants to import it here, but we always prefer to use the external path here.

        for dep in &db.crate_graph()[ctx.from.krate].dependencies {
            let import_map = db.import_map(dep.crate_id);
            let Some(import_info_for) = import_map.import_info_for(item) else { continue };
            for info in import_info_for {
                if info.is_doc_hidden {
                    // the item or import is `#[doc(hidden)]`, so skip it as it is in an external crate
                    continue;
                }

                // Determine best path for containing module and append last segment from `info`.
                // FIXME: we should guide this to look up the path locally, or from the same crate again?
                let path =
                    find_path_for_module(ctx, visited_modules, info.container, best_path_len - 1);
                let Some((path, path_stability)) = path else {
                    continue;
                };
                cov_mark::hit!(partially_imported);
                let path = (
                    path,
                    zip_stability(path_stability, if info.is_unstable { Unstable } else { Stable }),
                );

                process(path, info.name.clone(), &mut best_path_len);
            }
        }
    }
    best_path
}

/// Select the best (most relevant) path between two paths.
/// This accounts for stability, path length whether, std should be chosen over alloc/core paths as
/// well as ignoring prelude like paths or not.
fn select_best_path(
    old_path @ (_, old_stability): (ModPath, Stability),
    new_path @ (_, new_stability): (ModPath, Stability),
    cfg: ImportPathConfig,
) -> (ModPath, Stability) {
    match (old_stability, new_stability) {
        (Stable, Unstable) => return old_path,
        (Unstable, Stable) => return new_path,
        _ => {}
    }
    const STD_CRATES: [Name; 3] = [known::std, known::core, known::alloc];

    let choose = |new: (ModPath, _), old: (ModPath, _)| {
        let (new_path, _) = &new;
        let (old_path, _) = &old;
        let new_has_prelude = new_path.segments().iter().any(|seg| seg == &known::prelude);
        let old_has_prelude = old_path.segments().iter().any(|seg| seg == &known::prelude);
        match (new_has_prelude, old_has_prelude, cfg.prefer_prelude) {
            (true, false, true) | (false, true, false) => new,
            (true, false, false) | (false, true, true) => old,
            // no prelude difference in the paths, so pick the shorter one
            (true, true, _) | (false, false, _) => {
                let new_path_is_shorter = new_path
                    .len()
                    .cmp(&old_path.len())
                    .then_with(|| new_path.textual_len().cmp(&old_path.textual_len()))
                    .is_lt();
                if new_path_is_shorter {
                    new
                } else {
                    old
                }
            }
        }
    };

    match (old_path.0.segments().first(), new_path.0.segments().first()) {
        (Some(old), Some(new)) if STD_CRATES.contains(old) && STD_CRATES.contains(new) => {
            let rank = match cfg.prefer_no_std {
                false => |name: &Name| match name {
                    name if name == &known::core => 0,
                    name if name == &known::alloc => 1,
                    name if name == &known::std => 2,
                    _ => unreachable!(),
                },
                true => |name: &Name| match name {
                    name if name == &known::core => 2,
                    name if name == &known::alloc => 1,
                    name if name == &known::std => 0,
                    _ => unreachable!(),
                },
            };
            let nrank = rank(new);
            let orank = rank(old);
            match nrank.cmp(&orank) {
                Ordering::Less => old_path,
                Ordering::Equal => choose(new_path, old_path),
                Ordering::Greater => new_path,
            }
        }
        _ => choose(new_path, old_path),
    }
}

/// Finds locations in `from.krate` from which `item` can be imported by `from`.
fn find_local_import_locations(
    db: &dyn DefDatabase,
    item: ItemInNs,
    from: ModuleId,
    def_map: &DefMap,
    mut cb: impl FnMut(&Name, ModuleId),
) {
    let _p = tracing::info_span!("find_local_import_locations").entered();

    // `from` can import anything below `from` with visibility of at least `from`, and anything
    // above `from` with any visibility. That means we do not need to descend into private siblings
    // of `from` (and similar).

    // Compute the initial worklist. We start with all direct child modules of `from` as well as all
    // of its (recursive) parent modules.
    let mut worklist = def_map[from.local_id]
        .children
        .values()
        .map(|child| def_map.module_id(*child))
        // FIXME: do we need to traverse out of block expressions here?
        .chain(iter::successors(from.containing_module(db), |m| m.containing_module(db)))
        .collect::<Vec<_>>();
    let mut seen: FxHashSet<_> = FxHashSet::default();

    let def_map = def_map.crate_root().def_map(db);

    while let Some(module) = worklist.pop() {
        if !seen.insert(module) {
            continue; // already processed this module
        }
        let ext_def_map;
        let data = if module.krate == from.krate {
            if module.block.is_some() {
                // Re-query the block's DefMap
                ext_def_map = module.def_map(db);
                &ext_def_map[module.local_id]
            } else {
                // Reuse the root DefMap
                &def_map[module.local_id]
            }
        } else {
            // The crate might reexport a module defined in another crate.
            ext_def_map = module.def_map(db);
            &ext_def_map[module.local_id]
        };

        if let Some((name, vis, declared)) = data.scope.name_of(item) {
            if vis.is_visible_from(db, from) {
                let is_pub_or_explicit = match vis {
                    Visibility::Module(_, VisibilityExplicitness::Explicit) => {
                        cov_mark::hit!(explicit_private_imports);
                        true
                    }
                    Visibility::Module(_, VisibilityExplicitness::Implicit) => {
                        cov_mark::hit!(discount_private_imports);
                        false
                    }
                    Visibility::Public => true,
                };

                // Ignore private imports unless they are explicit. these could be used if we are
                // in a submodule of this module, but that's usually not
                // what the user wants; and if this module can import
                // the item and we're a submodule of it, so can we.
                // Also this keeps the cached data smaller.
                if declared || is_pub_or_explicit {
                    cb(name, module);
                }
            }
        }

        // Descend into all modules visible from `from`.
        for (module, vis) in data.scope.modules_in_scope() {
            if vis.is_visible_from(db, from) {
                worklist.push(module);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use hir_expand::db::ExpandDatabase;
    use itertools::Itertools;
    use stdx::format_to;
    use syntax::ast::AstNode;
    use test_fixture::WithFixture;

    use crate::test_db::TestDB;

    use super::*;

    /// `code` needs to contain a cursor marker; checks that `find_path` for the
    /// item the `path` refers to returns that same path when called from the
    /// module the cursor is in.
    #[track_caller]
    fn check_found_path_(ra_fixture: &str, path: &str, prefer_prelude: bool, expect: Expect) {
        let (db, pos) = TestDB::with_position(ra_fixture);
        let module = db.module_at_position(pos);
        let parsed_path_file =
            syntax::SourceFile::parse(&format!("use {path};"), span::Edition::CURRENT);
        let ast_path =
            parsed_path_file.syntax_node().descendants().find_map(syntax::ast::Path::cast).unwrap();
        let mod_path = ModPath::from_src(&db, ast_path, &mut |range| {
            db.span_map(pos.file_id.into()).as_ref().span_for_range(range).ctx
        })
        .unwrap();

        let def_map = module.def_map(&db);
        let resolved = def_map
            .resolve_path(
                &db,
                module.local_id,
                &mod_path,
                crate::item_scope::BuiltinShadowMode::Module,
                None,
            )
            .0;
        let resolved = resolved
            .take_types()
            .map(ItemInNs::Types)
            .or_else(|| resolved.take_values().map(ItemInNs::Values))
            .expect("path does not resolve to a type or value");

        let mut res = String::new();
        for (prefix, ignore_local_imports) in
            [PrefixKind::Plain, PrefixKind::ByCrate, PrefixKind::BySelf]
                .into_iter()
                .cartesian_product([false, true])
        {
            let found_path = find_path(
                &db,
                resolved,
                module,
                prefix,
                ignore_local_imports,
                ImportPathConfig { prefer_no_std: false, prefer_prelude },
            );
            format_to!(
                res,
                "{:7}(imports {}): {}\n",
                format!("{:?}", prefix),
                if ignore_local_imports { '✖' } else { '✔' },
                found_path
                    .map_or_else(|| "<unresolvable>".to_owned(), |it| it.display(&db).to_string()),
            );
        }
        expect.assert_eq(&res);
    }

    fn check_found_path(ra_fixture: &str, path: &str, expect: Expect) {
        check_found_path_(ra_fixture, path, false, expect);
    }

    fn check_found_path_prelude(ra_fixture: &str, path: &str, expect: Expect) {
        check_found_path_(ra_fixture, path, true, expect);
    }

    #[test]
    fn same_module() {
        check_found_path(
            r#"
struct S;
$0
        "#,
            "S",
            expect![[r#"
                Plain  (imports ✔): S
                Plain  (imports ✖): S
                ByCrate(imports ✔): crate::S
                ByCrate(imports ✖): crate::S
                BySelf (imports ✔): self::S
                BySelf (imports ✖): self::S
            "#]],
        );
    }

    #[test]
    fn enum_variant() {
        check_found_path(
            r#"
enum E { A }
$0
        "#,
            "E::A",
            expect![[r#"
                Plain  (imports ✔): E::A
                Plain  (imports ✖): E::A
                ByCrate(imports ✔): crate::E::A
                ByCrate(imports ✖): crate::E::A
                BySelf (imports ✔): self::E::A
                BySelf (imports ✖): self::E::A
            "#]],
        );
    }

    #[test]
    fn sub_module() {
        check_found_path(
            r#"
mod foo {
    pub struct S;
}
$0
        "#,
            "foo::S",
            expect![[r#"
                Plain  (imports ✔): foo::S
                Plain  (imports ✖): foo::S
                ByCrate(imports ✔): crate::foo::S
                ByCrate(imports ✖): crate::foo::S
                BySelf (imports ✔): self::foo::S
                BySelf (imports ✖): self::foo::S
            "#]],
        );
    }

    #[test]
    fn super_module() {
        check_found_path(
            r#"
//- /main.rs
mod foo;
//- /foo.rs
mod bar;
struct S;
//- /foo/bar.rs
$0
        "#,
            "super::S",
            expect![[r#"
                Plain  (imports ✔): super::S
                Plain  (imports ✖): super::S
                ByCrate(imports ✔): crate::foo::S
                ByCrate(imports ✖): crate::foo::S
                BySelf (imports ✔): super::S
                BySelf (imports ✖): super::S
            "#]],
        );
    }

    #[test]
    fn self_module() {
        check_found_path(
            r#"
//- /main.rs
mod foo;
//- /foo.rs
$0
        "#,
            "self",
            expect![[r#"
                Plain  (imports ✔): self
                Plain  (imports ✖): self
                ByCrate(imports ✔): crate::foo
                ByCrate(imports ✖): crate::foo
                BySelf (imports ✔): self
                BySelf (imports ✖): self
            "#]],
        );
    }

    #[test]
    fn crate_root() {
        check_found_path(
            r#"
//- /main.rs
mod foo;
//- /foo.rs
$0
        "#,
            "crate",
            expect![[r#"
                Plain  (imports ✔): crate
                Plain  (imports ✖): crate
                ByCrate(imports ✔): crate
                ByCrate(imports ✖): crate
                BySelf (imports ✔): crate
                BySelf (imports ✖): crate
            "#]],
        );
    }

    #[test]
    fn same_crate() {
        check_found_path(
            r#"
//- /main.rs
mod foo;
struct S;
//- /foo.rs
$0
        "#,
            "crate::S",
            expect![[r#"
                Plain  (imports ✔): crate::S
                Plain  (imports ✖): crate::S
                ByCrate(imports ✔): crate::S
                ByCrate(imports ✖): crate::S
                BySelf (imports ✔): crate::S
                BySelf (imports ✖): crate::S
            "#]],
        );
    }

    #[test]
    fn different_crate() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:std
$0
//- /std.rs crate:std
pub struct S;
        "#,
            "std::S",
            expect![[r#"
                Plain  (imports ✔): std::S
                Plain  (imports ✖): std::S
                ByCrate(imports ✔): std::S
                ByCrate(imports ✖): std::S
                BySelf (imports ✔): std::S
                BySelf (imports ✖): std::S
            "#]],
        );
    }

    #[test]
    fn different_crate_renamed() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:std
extern crate std as std_renamed;
$0
//- /std.rs crate:std
pub struct S;
        "#,
            "std_renamed::S",
            expect![[r#"
                Plain  (imports ✔): std_renamed::S
                Plain  (imports ✖): std_renamed::S
                ByCrate(imports ✔): std_renamed::S
                ByCrate(imports ✖): std_renamed::S
                BySelf (imports ✔): std_renamed::S
                BySelf (imports ✖): std_renamed::S
            "#]],
        );
    }

    #[test]
    fn partially_imported() {
        cov_mark::check!(partially_imported);
        // Tests that short paths are used even for external items, when parts of the path are
        // already in scope.
        check_found_path(
            r#"
//- /main.rs crate:main deps:syntax

use syntax::ast;
$0

//- /lib.rs crate:syntax
pub mod ast {
    pub enum ModuleItem {
        A, B, C,
    }
}
        "#,
            "syntax::ast::ModuleItem",
            expect![[r#"
                Plain  (imports ✔): ast::ModuleItem
                Plain  (imports ✖): syntax::ast::ModuleItem
                ByCrate(imports ✔): crate::ast::ModuleItem
                ByCrate(imports ✖): syntax::ast::ModuleItem
                BySelf (imports ✔): self::ast::ModuleItem
                BySelf (imports ✖): syntax::ast::ModuleItem
            "#]],
        );

        check_found_path(
            r#"
//- /main.rs crate:main deps:syntax
$0

//- /lib.rs crate:syntax
pub mod ast {
    pub enum ModuleItem {
        A, B, C,
    }
}
        "#,
            "syntax::ast::ModuleItem",
            expect![[r#"
                Plain  (imports ✔): syntax::ast::ModuleItem
                Plain  (imports ✖): syntax::ast::ModuleItem
                ByCrate(imports ✔): syntax::ast::ModuleItem
                ByCrate(imports ✖): syntax::ast::ModuleItem
                BySelf (imports ✔): syntax::ast::ModuleItem
                BySelf (imports ✖): syntax::ast::ModuleItem
            "#]],
        );
    }

    #[test]
    fn same_crate_reexport() {
        check_found_path(
            r#"
mod bar {
    mod foo { pub(super) struct S; }
    pub(crate) use foo::*;
}
$0
        "#,
            "bar::S",
            expect![[r#"
                Plain  (imports ✔): bar::S
                Plain  (imports ✖): bar::S
                ByCrate(imports ✔): crate::bar::S
                ByCrate(imports ✖): crate::bar::S
                BySelf (imports ✔): self::bar::S
                BySelf (imports ✖): self::bar::S
            "#]],
        );
    }

    #[test]
    fn same_crate_reexport_rename() {
        check_found_path(
            r#"
mod bar {
    mod foo { pub(super) struct S; }
    pub(crate) use foo::S as U;
}
$0
        "#,
            "bar::U",
            expect![[r#"
                Plain  (imports ✔): bar::U
                Plain  (imports ✖): bar::U
                ByCrate(imports ✔): crate::bar::U
                ByCrate(imports ✖): crate::bar::U
                BySelf (imports ✔): self::bar::U
                BySelf (imports ✖): self::bar::U
            "#]],
        );
    }

    #[test]
    fn different_crate_reexport() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:std
$0
//- /std.rs crate:std deps:core
pub use core::S;
//- /core.rs crate:core
pub struct S;
        "#,
            "std::S",
            expect![[r#"
                Plain  (imports ✔): std::S
                Plain  (imports ✖): std::S
                ByCrate(imports ✔): std::S
                ByCrate(imports ✖): std::S
                BySelf (imports ✔): std::S
                BySelf (imports ✖): std::S
            "#]],
        );
    }

    #[test]
    fn prelude() {
        check_found_path(
            r#"
//- /main.rs edition:2018 crate:main deps:std
$0
//- /std.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub struct S;
    }
}
        "#,
            "S",
            expect![[r#"
                Plain  (imports ✔): S
                Plain  (imports ✖): S
                ByCrate(imports ✔): S
                ByCrate(imports ✖): S
                BySelf (imports ✔): S
                BySelf (imports ✖): S
            "#]],
        );
    }

    #[test]
    fn shadowed_prelude() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:std
struct S;
$0
//- /std.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub struct S;
    }
}
"#,
            "std::prelude::rust_2018::S",
            expect![[r#"
                Plain  (imports ✔): std::prelude::rust_2018::S
                Plain  (imports ✖): std::prelude::rust_2018::S
                ByCrate(imports ✔): std::prelude::rust_2018::S
                ByCrate(imports ✖): std::prelude::rust_2018::S
                BySelf (imports ✔): std::prelude::rust_2018::S
                BySelf (imports ✖): std::prelude::rust_2018::S
            "#]],
        );
    }

    #[test]
    fn imported_prelude() {
        check_found_path(
            r#"
//- /main.rs edition:2018 crate:main deps:std
use S;
$0
//- /std.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub struct S;
    }
}
"#,
            "S",
            expect![[r#"
                Plain  (imports ✔): S
                Plain  (imports ✖): S
                ByCrate(imports ✔): crate::S
                ByCrate(imports ✖): S
                BySelf (imports ✔): self::S
                BySelf (imports ✖): S
            "#]],
        );
    }

    #[test]
    fn enum_variant_from_prelude() {
        let code = r#"
//- /main.rs edition:2018 crate:main deps:std
$0
//- /std.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub enum Option<T> { Some(T), None }
        pub use Option::*;
    }
}
        "#;
        check_found_path(
            code,
            "None",
            expect![[r#"
                Plain  (imports ✔): None
                Plain  (imports ✖): None
                ByCrate(imports ✔): None
                ByCrate(imports ✖): None
                BySelf (imports ✔): None
                BySelf (imports ✖): None
            "#]],
        );
        check_found_path(
            code,
            "Some",
            expect![[r#"
                Plain  (imports ✔): Some
                Plain  (imports ✖): Some
                ByCrate(imports ✔): Some
                ByCrate(imports ✖): Some
                BySelf (imports ✔): Some
                BySelf (imports ✖): Some
            "#]],
        );
    }

    #[test]
    fn shortest_path() {
        check_found_path(
            r#"
//- /main.rs
pub mod foo;
pub mod baz;
struct S;
$0
//- /foo.rs
pub mod bar { pub struct S; }
//- /baz.rs
pub use crate::foo::bar::S;
        "#,
            "baz::S",
            expect![[r#"
                Plain  (imports ✔): baz::S
                Plain  (imports ✖): baz::S
                ByCrate(imports ✔): crate::baz::S
                ByCrate(imports ✖): crate::baz::S
                BySelf (imports ✔): self::baz::S
                BySelf (imports ✖): self::baz::S
            "#]],
        );
    }

    #[test]
    fn discount_private_imports() {
        cov_mark::check!(discount_private_imports);
        check_found_path(
            r#"
//- /main.rs
mod foo;
pub mod bar { pub struct S; }
use bar::S;
//- /foo.rs
$0
        "#,
            // crate::S would be shorter, but using private imports seems wrong
            "crate::bar::S",
            expect![[r#"
                Plain  (imports ✔): crate::bar::S
                Plain  (imports ✖): crate::bar::S
                ByCrate(imports ✔): crate::bar::S
                ByCrate(imports ✖): crate::bar::S
                BySelf (imports ✔): crate::bar::S
                BySelf (imports ✖): crate::bar::S
            "#]],
        );
    }

    #[test]
    fn explicit_private_imports_crate() {
        cov_mark::check!(explicit_private_imports);
        check_found_path(
            r#"
//- /main.rs
mod foo;
pub mod bar { pub struct S; }
pub(crate) use bar::S;
//- /foo.rs
$0
        "#,
            "crate::S",
            expect![[r#"
                Plain  (imports ✔): crate::S
                Plain  (imports ✖): crate::S
                ByCrate(imports ✔): crate::S
                ByCrate(imports ✖): crate::S
                BySelf (imports ✔): crate::S
                BySelf (imports ✖): crate::S
            "#]],
        );
    }

    #[test]
    fn explicit_private_imports() {
        cov_mark::check!(explicit_private_imports);
        check_found_path(
            r#"
//- /main.rs
pub mod bar {
    mod foo;
    pub mod baz { pub struct S; }
    pub(self) use baz::S;
}

//- /bar/foo.rs
$0
        "#,
            "super::S",
            expect![[r#"
                Plain  (imports ✔): super::S
                Plain  (imports ✖): super::S
                ByCrate(imports ✔): crate::bar::S
                ByCrate(imports ✖): crate::bar::S
                BySelf (imports ✔): super::S
                BySelf (imports ✖): super::S
            "#]],
        );
    }

    #[test]
    fn import_cycle() {
        check_found_path(
            r#"
//- /main.rs
pub mod foo;
pub mod bar;
pub mod baz;
//- /bar.rs
$0
//- /foo.rs
pub use super::baz;
pub struct S;
//- /baz.rs
pub use super::foo;
        "#,
            "crate::foo::S",
            expect![[r#"
                Plain  (imports ✔): crate::foo::S
                Plain  (imports ✖): crate::foo::S
                ByCrate(imports ✔): crate::foo::S
                ByCrate(imports ✖): crate::foo::S
                BySelf (imports ✔): crate::foo::S
                BySelf (imports ✖): crate::foo::S
            "#]],
        );
    }

    #[test]
    fn prefer_std_paths_over_alloc() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:alloc,std
$0

//- /std.rs crate:std deps:alloc
pub mod sync {
    pub use alloc::sync::Arc;
}

//- /zzz.rs crate:alloc
pub mod sync {
    pub struct Arc;
}
        "#,
            "std::sync::Arc",
            expect![[r#"
                Plain  (imports ✔): std::sync::Arc
                Plain  (imports ✖): std::sync::Arc
                ByCrate(imports ✔): std::sync::Arc
                ByCrate(imports ✖): std::sync::Arc
                BySelf (imports ✔): std::sync::Arc
                BySelf (imports ✖): std::sync::Arc
            "#]],
        );
    }

    #[test]
    fn prefer_core_paths_over_std() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:core,std
#![no_std]

$0

//- /std.rs crate:std deps:core

pub mod fmt {
    pub use core::fmt::Error;
}

//- /zzz.rs crate:core

pub mod fmt {
    pub struct Error;
}
        "#,
            "core::fmt::Error",
            expect![[r#"
                Plain  (imports ✔): core::fmt::Error
                Plain  (imports ✖): core::fmt::Error
                ByCrate(imports ✔): core::fmt::Error
                ByCrate(imports ✖): core::fmt::Error
                BySelf (imports ✔): core::fmt::Error
                BySelf (imports ✖): core::fmt::Error
            "#]],
        );

        // Should also work (on a best-effort basis) if `no_std` is conditional.
        check_found_path(
            r#"
//- /main.rs crate:main deps:core,std
#![cfg_attr(not(test), no_std)]

$0

//- /std.rs crate:std deps:core

pub mod fmt {
    pub use core::fmt::Error;
}

//- /zzz.rs crate:core

pub mod fmt {
    pub struct Error;
}
        "#,
            "core::fmt::Error",
            expect![[r#"
                Plain  (imports ✔): core::fmt::Error
                Plain  (imports ✖): core::fmt::Error
                ByCrate(imports ✔): core::fmt::Error
                ByCrate(imports ✖): core::fmt::Error
                BySelf (imports ✔): core::fmt::Error
                BySelf (imports ✖): core::fmt::Error
            "#]],
        );
    }

    #[test]
    fn prefer_alloc_paths_over_std() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:alloc,std
#![no_std]

extern crate alloc;

$0

//- /std.rs crate:std deps:alloc

pub mod sync {
    pub use alloc::sync::Arc;
}

//- /zzz.rs crate:alloc

pub mod sync {
    pub struct Arc;
}
            "#,
            "alloc::sync::Arc",
            expect![[r#"
                Plain  (imports ✔): alloc::sync::Arc
                Plain  (imports ✖): alloc::sync::Arc
                ByCrate(imports ✔): alloc::sync::Arc
                ByCrate(imports ✖): alloc::sync::Arc
                BySelf (imports ✔): alloc::sync::Arc
                BySelf (imports ✖): alloc::sync::Arc
            "#]],
        );
    }

    #[test]
    fn prefer_shorter_paths_if_not_alloc() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:megaalloc,std
$0

//- /std.rs crate:std deps:megaalloc
pub mod sync {
    pub use megaalloc::sync::Arc;
}

//- /zzz.rs crate:megaalloc
pub struct Arc;
            "#,
            "megaalloc::Arc",
            expect![[r#"
                Plain  (imports ✔): megaalloc::Arc
                Plain  (imports ✖): megaalloc::Arc
                ByCrate(imports ✔): megaalloc::Arc
                ByCrate(imports ✖): megaalloc::Arc
                BySelf (imports ✔): megaalloc::Arc
                BySelf (imports ✖): megaalloc::Arc
            "#]],
        );
    }

    #[test]
    fn builtins_are_in_scope() {
        let code = r#"
$0

pub mod primitive {
    pub use u8;
}
        "#;
        check_found_path(
            code,
            "u8",
            expect![[r#"
                Plain  (imports ✔): u8
                Plain  (imports ✖): u8
                ByCrate(imports ✔): u8
                ByCrate(imports ✖): u8
                BySelf (imports ✔): u8
                BySelf (imports ✖): u8
            "#]],
        );
        check_found_path(
            code,
            "u16",
            expect![[r#"
                Plain  (imports ✔): u16
                Plain  (imports ✖): u16
                ByCrate(imports ✔): u16
                ByCrate(imports ✖): u16
                BySelf (imports ✔): u16
                BySelf (imports ✖): u16
            "#]],
        );
    }

    #[test]
    fn inner_items() {
        check_found_path(
            r#"
fn main() {
    struct Inner {}
    $0
}
        "#,
            "Inner",
            expect![[r#"
                Plain  (imports ✔): Inner
                Plain  (imports ✖): Inner
                ByCrate(imports ✔): Inner
                ByCrate(imports ✖): Inner
                BySelf (imports ✔): Inner
                BySelf (imports ✖): Inner
            "#]],
        );
    }

    #[test]
    fn inner_items_from_outer_scope() {
        check_found_path(
            r#"
fn main() {
    struct Struct {}
    {
        $0
    }
}
        "#,
            "Struct",
            expect![[r#"
                Plain  (imports ✔): Struct
                Plain  (imports ✖): Struct
                ByCrate(imports ✔): Struct
                ByCrate(imports ✖): Struct
                BySelf (imports ✔): Struct
                BySelf (imports ✖): Struct
            "#]],
        );
    }

    #[test]
    fn inner_items_from_inner_module() {
        check_found_path(
            r#"
fn main() {
    mod module {
        pub struct Struct {}
    }
    {
        $0
    }
}
        "#,
            "module::Struct",
            expect![[r#"
                Plain  (imports ✔): module::Struct
                Plain  (imports ✖): module::Struct
                ByCrate(imports ✔): module::Struct
                ByCrate(imports ✖): module::Struct
                BySelf (imports ✔): module::Struct
                BySelf (imports ✖): module::Struct
            "#]],
        );
    }

    #[test]
    fn outer_items_with_inner_items_present() {
        check_found_path(
            r#"
mod module {
    pub struct CompleteMe;
}

fn main() {
    fn inner() {}
    $0
}
            "#,
            "module::CompleteMe",
            expect![[r#"
                Plain  (imports ✔): module::CompleteMe
                Plain  (imports ✖): module::CompleteMe
                ByCrate(imports ✔): crate::module::CompleteMe
                ByCrate(imports ✖): crate::module::CompleteMe
                BySelf (imports ✔): self::module::CompleteMe
                BySelf (imports ✖): self::module::CompleteMe
            "#]],
        )
    }

    #[test]
    fn from_inside_module() {
        // This worked correctly, but the test suite logic was broken.
        cov_mark::check!(submodule_in_testdb);
        check_found_path(
            r#"
mod baz {
    pub struct Foo {}
}

mod bar {
    fn bar() {
        $0
    }
}
            "#,
            "crate::baz::Foo",
            expect![[r#"
                Plain  (imports ✔): crate::baz::Foo
                Plain  (imports ✖): crate::baz::Foo
                ByCrate(imports ✔): crate::baz::Foo
                ByCrate(imports ✖): crate::baz::Foo
                BySelf (imports ✔): crate::baz::Foo
                BySelf (imports ✖): crate::baz::Foo
            "#]],
        )
    }

    #[test]
    fn from_inside_module_with_inner_items() {
        check_found_path(
            r#"
mod baz {
    pub struct Foo {}
}

mod bar {
    fn bar() {
        fn inner() {}
        $0
    }
}
            "#,
            "crate::baz::Foo",
            expect![[r#"
                Plain  (imports ✔): crate::baz::Foo
                Plain  (imports ✖): crate::baz::Foo
                ByCrate(imports ✔): crate::baz::Foo
                ByCrate(imports ✖): crate::baz::Foo
                BySelf (imports ✔): crate::baz::Foo
                BySelf (imports ✖): crate::baz::Foo
            "#]],
        )
    }

    #[test]
    fn recursive_pub_mod_reexport() {
        check_found_path(
            r#"
fn main() {
    let _ = 22_i32.as_name$0();
}

pub mod name {
    pub trait AsName {
        fn as_name(&self) -> String;
    }
    impl AsName for i32 {
        fn as_name(&self) -> String {
            format!("Name: {}", self)
        }
    }
    pub use crate::name;
}
"#,
            "name::AsName",
            expect![[r#"
                Plain  (imports ✔): name::AsName
                Plain  (imports ✖): name::AsName
                ByCrate(imports ✔): crate::name::AsName
                ByCrate(imports ✖): crate::name::AsName
                BySelf (imports ✔): self::name::AsName
                BySelf (imports ✖): self::name::AsName
            "#]],
        );
    }

    #[test]
    fn extern_crate() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:dep
$0
//- /dep.rs crate:dep
"#,
            "dep",
            expect![[r#"
                Plain  (imports ✔): dep
                Plain  (imports ✖): dep
                ByCrate(imports ✔): dep
                ByCrate(imports ✖): dep
                BySelf (imports ✔): dep
                BySelf (imports ✖): dep
            "#]],
        );

        check_found_path(
            r#"
//- /main.rs crate:main deps:dep
fn f() {
    fn inner() {}
    $0
}
//- /dep.rs crate:dep
"#,
            "dep",
            expect![[r#"
                Plain  (imports ✔): dep
                Plain  (imports ✖): dep
                ByCrate(imports ✔): dep
                ByCrate(imports ✖): dep
                BySelf (imports ✔): dep
                BySelf (imports ✖): dep
            "#]],
        );
    }

    #[test]
    fn prelude_with_inner_items() {
        check_found_path(
            r#"
//- /main.rs edition:2018 crate:main deps:std
fn f() {
    fn inner() {}
    $0
}
//- /std.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub enum Option { None }
        pub use Option::*;
    }
}
        "#,
            "None",
            expect![[r#"
                Plain  (imports ✔): None
                Plain  (imports ✖): None
                ByCrate(imports ✔): None
                ByCrate(imports ✖): None
                BySelf (imports ✔): None
                BySelf (imports ✖): None
            "#]],
        );
    }

    #[test]
    fn different_crate_renamed_through_dep() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:intermediate
$0
//- /intermediate.rs crate:intermediate deps:std
pub extern crate std as std_renamed;
//- /std.rs crate:std
pub struct S;
    "#,
            "intermediate::std_renamed::S",
            expect![[r#"
                Plain  (imports ✔): intermediate::std_renamed::S
                Plain  (imports ✖): intermediate::std_renamed::S
                ByCrate(imports ✔): intermediate::std_renamed::S
                ByCrate(imports ✖): intermediate::std_renamed::S
                BySelf (imports ✔): intermediate::std_renamed::S
                BySelf (imports ✖): intermediate::std_renamed::S
            "#]],
        );
    }

    #[test]
    fn different_crate_doc_hidden() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:intermediate
$0
//- /intermediate.rs crate:intermediate deps:std
#[doc(hidden)]
pub extern crate std;
pub extern crate std as longer;
//- /std.rs crate:std
pub struct S;
    "#,
            "intermediate::longer::S",
            expect![[r#"
                Plain  (imports ✔): intermediate::longer::S
                Plain  (imports ✖): intermediate::longer::S
                ByCrate(imports ✔): intermediate::longer::S
                ByCrate(imports ✖): intermediate::longer::S
                BySelf (imports ✔): intermediate::longer::S
                BySelf (imports ✖): intermediate::longer::S
            "#]],
        );
    }

    #[test]
    fn respect_doc_hidden() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:std,lazy_static
$0
//- /lazy_static.rs crate:lazy_static deps:core
#[doc(hidden)]
pub use core::ops::Deref as __Deref;
//- /std.rs crate:std deps:core
pub use core::ops;
//- /core.rs crate:core
pub mod ops {
    pub trait Deref {}
}
    "#,
            "std::ops::Deref",
            expect![[r#"
                Plain  (imports ✔): std::ops::Deref
                Plain  (imports ✖): std::ops::Deref
                ByCrate(imports ✔): std::ops::Deref
                ByCrate(imports ✖): std::ops::Deref
                BySelf (imports ✔): std::ops::Deref
                BySelf (imports ✖): std::ops::Deref
            "#]],
        );
    }

    #[test]
    fn respect_unstable_modules() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:std,core
#![no_std]
extern crate std;
$0
//- /longer.rs crate:std deps:core
pub mod error {
    pub use core::error::Error;
}
//- /core.rs crate:core
pub mod error {
    #![unstable(feature = "error_in_core", issue = "103765")]
    pub trait Error {}
}
"#,
            "std::error::Error",
            expect![[r#"
                Plain  (imports ✔): std::error::Error
                Plain  (imports ✖): std::error::Error
                ByCrate(imports ✔): std::error::Error
                ByCrate(imports ✖): std::error::Error
                BySelf (imports ✔): std::error::Error
                BySelf (imports ✖): std::error::Error
            "#]],
        );
    }

    #[test]
    fn respects_prelude_setting() {
        let ra_fixture = r#"
//- /main.rs crate:main deps:krate
$0
//- /krate.rs crate:krate
pub mod prelude {
    pub use crate::foo::*;
}

pub mod foo {
    pub struct Foo;
}
"#;
        check_found_path(
            ra_fixture,
            "krate::foo::Foo",
            expect![[r#"
                Plain  (imports ✔): krate::foo::Foo
                Plain  (imports ✖): krate::foo::Foo
                ByCrate(imports ✔): krate::foo::Foo
                ByCrate(imports ✖): krate::foo::Foo
                BySelf (imports ✔): krate::foo::Foo
                BySelf (imports ✖): krate::foo::Foo
            "#]],
        );
        check_found_path_prelude(
            ra_fixture,
            "krate::prelude::Foo",
            expect![[r#"
                Plain  (imports ✔): krate::prelude::Foo
                Plain  (imports ✖): krate::prelude::Foo
                ByCrate(imports ✔): krate::prelude::Foo
                ByCrate(imports ✖): krate::prelude::Foo
                BySelf (imports ✔): krate::prelude::Foo
                BySelf (imports ✖): krate::prelude::Foo
            "#]],
        );
    }

    #[test]
    fn respect_segment_length() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:petgraph
$0
//- /petgraph.rs crate:petgraph
pub mod graph {
    pub use crate::graph_impl::{
        NodeIndex
    };
}

mod graph_impl {
    pub struct NodeIndex<Ix>(Ix);
}

pub mod stable_graph {
    #[doc(no_inline)]
    pub use crate::graph::{NodeIndex};
}

pub mod prelude {
    #[doc(no_inline)]
    pub use crate::graph::{NodeIndex};
}
"#,
            "petgraph::graph::NodeIndex",
            expect![[r#"
                Plain  (imports ✔): petgraph::graph::NodeIndex
                Plain  (imports ✖): petgraph::graph::NodeIndex
                ByCrate(imports ✔): petgraph::graph::NodeIndex
                ByCrate(imports ✖): petgraph::graph::NodeIndex
                BySelf (imports ✔): petgraph::graph::NodeIndex
                BySelf (imports ✖): petgraph::graph::NodeIndex
            "#]],
        );
    }

    #[test]
    fn regression_17271() {
        check_found_path(
            r#"
//- /lib.rs crate:main
mod foo;

//- /foo.rs
mod bar;

pub fn b() {$0}
//- /foo/bar.rs
pub fn c() {}
"#,
            "bar::c",
            expect![[r#"
                Plain  (imports ✔): bar::c
                Plain  (imports ✖): bar::c
                ByCrate(imports ✔): crate::foo::bar::c
                ByCrate(imports ✖): crate::foo::bar::c
                BySelf (imports ✔): self::bar::c
                BySelf (imports ✖): self::bar::c
            "#]],
        );
    }
}
