//! An algorithm to find a path to refer to a certain item.

use std::{cell::Cell, cmp::Ordering, iter};

use base_db::{Crate, CrateOrigin, LangCrateOrigin};
use hir_expand::{
    Lookup,
    mod_path::{ModPath, PathKind},
    name::{AsName, Name},
};
use intern::sym;
use rustc_hash::FxHashSet;

use crate::{
    ImportPathConfig, ModuleDefId, ModuleId,
    db::DefDatabase,
    item_scope::ItemInNs,
    nameres::DefMap,
    visibility::{Visibility, VisibilityExplicitness},
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
    let item_module = item.module(db)?;
    if item_module.is_within_block() {
        prefix_kind = PrefixKind::Plain;
    }
    cfg.prefer_no_std = cfg.prefer_no_std || db.crate_supports_no_std(from.krate());

    find_path_inner(
        &FindPathCtx {
            db,
            prefix: prefix_kind,
            cfg,
            ignore_local_imports,
            is_std_item: item_module.krate().data(db).origin.is_lang(),
            from,
            from_def_map: from.def_map(db),
            fuel: Cell::new(FIND_PATH_FUEL),
        },
        item,
        MAX_PATH_LEN,
    )
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Stability {
    Unstable,
    Stable,
}
use Stability::*;

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
    is_std_item: bool,
    from: ModuleId,
    from_def_map: &'db DefMap,
    fuel: Cell<usize>,
}

/// Attempts to find a path to refer to the given `item` visible from the `from` ModuleId
fn find_path_inner(ctx: &FindPathCtx<'_>, item: ItemInNs, max_len: usize) -> Option<ModPath> {
    // - if the item is a module, jump straight to module search
    if !ctx.is_std_item
        && let ItemInNs::Types(ModuleDefId::ModuleId(module_id)) = item
    {
        return find_path_for_module(ctx, &mut FxHashSet::default(), module_id, true, max_len)
            .map(|choice| choice.path);
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
        return Some(value.path);
    }

    if let Some(ModuleDefId::EnumVariantId(variant)) = item.as_module_def_id() {
        // - if the item is an enum variant, refer to it via the enum
        let loc = variant.lookup(ctx.db);
        if let Some(mut path) = find_path_inner(ctx, ItemInNs::Types(loc.parent.into()), max_len) {
            path.push_segment(
                loc.parent.enum_variants(ctx.db).variants[loc.index as usize].1.clone(),
            );
            return Some(path);
        }
        // If this doesn't work, it seems we have no way of referring to the
        // enum; that's very weird, but there might still be a reexport of the
        // variant somewhere
    }

    let mut best_choice = None;
    calculate_best_path(ctx, &mut FxHashSet::default(), item, max_len, &mut best_choice);
    best_choice.map(|choice| choice.path)
}

#[tracing::instrument(skip_all)]
fn find_path_for_module(
    ctx: &FindPathCtx<'_>,
    visited_modules: &mut FxHashSet<(ItemInNs, ModuleId)>,
    module_id: ModuleId,
    maybe_extern: bool,
    max_len: usize,
) -> Option<Choice> {
    if max_len == 0 {
        // recursive base case, we can't find a path of length 0
        return None;
    }
    if let Some(crate_root) = module_id.as_crate_root() {
        if !maybe_extern || crate_root == ctx.from.derive_crate_root() {
            // - if the item is the crate root, return `crate`
            return Some(Choice {
                path: ModPath::from_segments(PathKind::Crate, None),
                path_text_len: 5,
                stability: Stable,
                prefer_due_to_prelude: false,
            });
        }
        // - otherwise if the item is the crate root of a dependency crate, return the name from the extern prelude

        let root_local_def_map = ctx.from.derive_crate_root().local_def_map(ctx.db).1;
        // rev here so we prefer looking at renamed extern decls first
        for (name, (def_id, _extern_crate)) in root_local_def_map.extern_prelude().rev() {
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
            } else if ctx.cfg.prefer_absolute {
                PathKind::Abs
            } else {
                PathKind::Plain
            };
            return Some(Choice::new(ctx.cfg.prefer_prelude, kind, name.clone(), Stable));
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
            return Some(Choice::new(
                ctx.cfg.prefer_prelude,
                ctx.prefix.path_kind(),
                scope_name,
                Stable,
            ));
        }
    }

    // - if the module can be referenced as self, super or crate, do that
    if let Some(kind) = is_kw_kind_relative_to_from(ctx.from_def_map, module_id, ctx.from)
        && (ctx.prefix != PrefixKind::ByCrate || kind == PathKind::Crate)
    {
        return Some(Choice {
            path: ModPath::from_segments(kind, None),
            path_text_len: path_kind_len(kind),
            stability: Stable,
            prefer_due_to_prelude: false,
        });
    }

    // - if the module is in the prelude, return it by that path
    let item = ItemInNs::Types(module_id.into());
    if let Some(choice) = find_in_prelude(ctx.db, ctx.from_def_map, item, ctx.from) {
        return Some(choice);
    }
    let mut best_choice = None;
    if maybe_extern {
        calculate_best_path(ctx, visited_modules, item, max_len, &mut best_choice);
    } else {
        calculate_best_path_local(ctx, visited_modules, item, max_len, &mut best_choice);
    }
    best_choice
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
) -> Option<Choice> {
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
        Some(Choice::new(false, PathKind::Plain, name.clone(), Stable))
    } else {
        None
    }
}

fn is_kw_kind_relative_to_from(
    def_map: &DefMap,
    item: ModuleId,
    from: ModuleId,
) -> Option<PathKind> {
    if item.krate != from.krate || item.is_within_block() || from.is_within_block() {
        return None;
    }
    let item = item.local_id;
    let from = from.local_id;
    if item == from {
        // - if the item is the module we're in, use `self`
        Some(PathKind::SELF)
    } else if let Some(parent_id) = def_map[from].parent {
        if item == parent_id {
            // - if the item is the parent module, use `super` (this is not used recursively, since `super::super` is ugly)
            Some(if parent_id == DefMap::ROOT { PathKind::Crate } else { PathKind::Super(1) })
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
    visited_modules: &mut FxHashSet<(ItemInNs, ModuleId)>,
    item: ItemInNs,
    max_len: usize,
    best_choice: &mut Option<Choice>,
) {
    let fuel = ctx.fuel.get();
    if fuel == 0 {
        // we ran out of fuel, so we stop searching here
        tracing::warn!(
            "ran out of fuel while searching for a path for item {item:?} of krate {:?} from krate {:?}",
            item.krate(ctx.db),
            ctx.from.krate()
        );
        return;
    }
    ctx.fuel.set(fuel - 1);

    if item.krate(ctx.db) == Some(ctx.from.krate) {
        // Item was defined in the same crate that wants to import it. It cannot be found in any
        // dependency in this case.
        calculate_best_path_local(ctx, visited_modules, item, max_len, best_choice)
    } else if ctx.is_std_item {
        // The item we are searching for comes from the sysroot libraries, so skip prefer looking in
        // the sysroot libraries directly.
        // We do need to fallback as the item in question could be re-exported by another crate
        // while not being a transitive dependency of the current crate.
        find_in_sysroot(ctx, visited_modules, item, max_len, best_choice)
    } else {
        // Item was defined in some upstream crate. This means that it must be exported from one,
        // too (unless we can't name it at all). It could *also* be (re)exported by the same crate
        // that wants to import it here, but we always prefer to use the external path here.

        ctx.from.krate.data(ctx.db).dependencies.iter().for_each(|dep| {
            find_in_dep(ctx, visited_modules, item, max_len, best_choice, dep.crate_id)
        });
    }
}

fn find_in_sysroot(
    ctx: &FindPathCtx<'_>,
    visited_modules: &mut FxHashSet<(ItemInNs, ModuleId)>,
    item: ItemInNs,
    max_len: usize,
    best_choice: &mut Option<Choice>,
) {
    let dependencies = &ctx.from.krate.data(ctx.db).dependencies;
    let mut search = |lang, best_choice: &mut _| {
        if let Some(dep) = dependencies.iter().filter(|it| it.is_sysroot()).find(|dep| {
            match dep.crate_id.data(ctx.db).origin {
                CrateOrigin::Lang(l) => l == lang,
                _ => false,
            }
        }) {
            find_in_dep(ctx, visited_modules, item, max_len, best_choice, dep.crate_id);
        }
    };
    if ctx.cfg.prefer_no_std {
        search(LangCrateOrigin::Core, best_choice);
        if matches!(best_choice, Some(Choice { stability: Stable, .. })) {
            return;
        }
        search(LangCrateOrigin::Std, best_choice);
        if matches!(best_choice, Some(Choice { stability: Stable, .. })) {
            return;
        }
    } else {
        search(LangCrateOrigin::Std, best_choice);
        if matches!(best_choice, Some(Choice { stability: Stable, .. })) {
            return;
        }
        search(LangCrateOrigin::Core, best_choice);
        if matches!(best_choice, Some(Choice { stability: Stable, .. })) {
            return;
        }
    }
    dependencies
        .iter()
        .filter(|it| it.is_sysroot())
        .chain(dependencies.iter().filter(|it| !it.is_sysroot()))
        .for_each(|dep| {
            find_in_dep(ctx, visited_modules, item, max_len, best_choice, dep.crate_id);
        });
}

fn find_in_dep(
    ctx: &FindPathCtx<'_>,
    visited_modules: &mut FxHashSet<(ItemInNs, ModuleId)>,
    item: ItemInNs,
    max_len: usize,
    best_choice: &mut Option<Choice>,
    dep: Crate,
) {
    let import_map = ctx.db.import_map(dep);
    let Some(import_info_for) = import_map.import_info_for(item) else {
        return;
    };
    for info in import_info_for {
        if info.is_doc_hidden {
            // the item or import is `#[doc(hidden)]`, so skip it as it is in an external crate
            continue;
        }

        // Determine best path for containing module and append last segment from `info`.
        // FIXME: we should guide this to look up the path locally, or from the same crate again?
        let choice = find_path_for_module(
            ctx,
            visited_modules,
            info.container,
            true,
            best_choice.as_ref().map_or(max_len, |it| it.path.len()) - 1,
        );
        let Some(mut choice) = choice else {
            continue;
        };
        cov_mark::hit!(partially_imported);
        if info.is_unstable {
            if !ctx.cfg.allow_unstable {
                // the item is unstable and we are not allowed to use unstable items
                continue;
            }
            choice.stability = Unstable;
        }

        Choice::try_select(best_choice, choice, ctx.cfg.prefer_prelude, info.name.clone());
    }
}

fn calculate_best_path_local(
    ctx: &FindPathCtx<'_>,
    visited_modules: &mut FxHashSet<(ItemInNs, ModuleId)>,
    item: ItemInNs,
    max_len: usize,
    best_choice: &mut Option<Choice>,
) {
    // FIXME: cache the `find_local_import_locations` output?
    find_local_import_locations(
        ctx.db,
        item,
        ctx.from,
        ctx.from_def_map,
        visited_modules,
        |visited_modules, name, module_id| {
            // we are looking for paths of length up to best_path_len, any longer will make it be
            // less optimal. The -1 is due to us pushing name onto it afterwards.
            if let Some(choice) = find_path_for_module(
                ctx,
                visited_modules,
                module_id,
                false,
                best_choice.as_ref().map_or(max_len, |it| it.path.len()) - 1,
            ) {
                Choice::try_select(best_choice, choice, ctx.cfg.prefer_prelude, name.clone());
            }
        },
    );
}

#[derive(Debug)]
struct Choice {
    path: ModPath,
    /// The length in characters of the path
    path_text_len: usize,
    /// The stability of the path
    stability: Stability,
    /// Whether this path contains a prelude segment and preference for it has been signaled
    prefer_due_to_prelude: bool,
}

impl Choice {
    fn new(prefer_prelude: bool, kind: PathKind, name: Name, stability: Stability) -> Self {
        Self {
            path_text_len: path_kind_len(kind) + name.as_str().len(),
            stability,
            prefer_due_to_prelude: prefer_prelude && name == sym::prelude,
            path: ModPath::from_segments(kind, iter::once(name)),
        }
    }

    fn push(mut self, prefer_prelude: bool, name: Name) -> Self {
        self.path_text_len += name.as_str().len();
        self.prefer_due_to_prelude |= prefer_prelude && name == sym::prelude;
        self.path.push_segment(name);
        self
    }

    fn try_select(
        current: &mut Option<Choice>,
        mut other: Choice,
        prefer_prelude: bool,
        name: Name,
    ) {
        let Some(current) = current else {
            *current = Some(other.push(prefer_prelude, name));
            return;
        };
        match other
            .stability
            .cmp(&current.stability)
            .then_with(|| other.prefer_due_to_prelude.cmp(&current.prefer_due_to_prelude))
            .then_with(|| (current.path.len()).cmp(&(other.path.len() + 1)))
        {
            Ordering::Less => return,
            Ordering::Equal => {
                other.path_text_len += name.as_str().len();
                if let Ordering::Less | Ordering::Equal =
                    current.path_text_len.cmp(&other.path_text_len)
                {
                    return;
                }
            }
            Ordering::Greater => {
                other.path_text_len += name.as_str().len();
            }
        }
        other.path.push_segment(name);
        *current = other;
    }
}

fn path_kind_len(kind: PathKind) -> usize {
    match kind {
        PathKind::Plain => 0,
        PathKind::Super(0) => 4,
        PathKind::Super(s) => s as usize * 5,
        PathKind::Crate => 5,
        PathKind::Abs => 2,
        PathKind::DollarCrate(_) => 0,
    }
}

/// Finds locations in `from.krate` from which `item` can be imported by `from`.
fn find_local_import_locations(
    db: &dyn DefDatabase,
    item: ItemInNs,
    from: ModuleId,
    def_map: &DefMap,
    visited_modules: &mut FxHashSet<(ItemInNs, ModuleId)>,
    mut cb: impl FnMut(&mut FxHashSet<(ItemInNs, ModuleId)>, &Name, ModuleId),
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
        .map(|&child| def_map.module_id(child))
        .chain(iter::successors(from.containing_module(db), |m| m.containing_module(db)))
        .zip(iter::repeat(false))
        .collect::<Vec<_>>();

    let def_map = def_map.crate_root().def_map(db);
    let mut block_def_map;
    let mut cursor = 0;

    while let Some(&mut (module, ref mut processed)) = worklist.get_mut(cursor) {
        cursor += 1;
        if !visited_modules.insert((item, module)) {
            // already processed this module
            continue;
        }
        *processed = true;
        let data = if module.block.is_some() {
            // Re-query the block's DefMap
            block_def_map = module.def_map(db);
            &block_def_map[module.local_id]
        } else {
            // Reuse the root DefMap
            &def_map[module.local_id]
        };

        if let Some((name, vis, declared)) = data.scope.name_of(item)
            && vis.is_visible_from(db, from)
        {
            let is_pub_or_explicit = match vis {
                Visibility::Module(_, VisibilityExplicitness::Explicit) => {
                    cov_mark::hit!(explicit_private_imports);
                    true
                }
                Visibility::Module(_, VisibilityExplicitness::Implicit) => {
                    cov_mark::hit!(discount_private_imports);
                    false
                }
                Visibility::PubCrate(_) => true,
                Visibility::Public => true,
            };

            // Ignore private imports unless they are explicit. these could be used if we are
            // in a submodule of this module, but that's usually not
            // what the user wants; and if this module can import
            // the item and we're a submodule of it, so can we.
            // Also this keeps the cached data smaller.
            if declared || is_pub_or_explicit {
                cb(visited_modules, name, module);
            }
        }

        // Descend into all modules visible from `from`.
        for (module, vis) in data.scope.modules_in_scope() {
            if module.krate != from.krate {
                // We don't need to look at modules from other crates as our item has to be in the
                // current crate
                continue;
            }
            if visited_modules.contains(&(item, module)) {
                continue;
            }

            if vis.is_visible_from(db, from) {
                worklist.push((module, false));
            }
        }
    }
    worklist.into_iter().filter(|&(_, processed)| processed).for_each(|(module, _)| {
        visited_modules.remove(&(item, module));
    });
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use hir_expand::db::ExpandDatabase;
    use itertools::Itertools;
    use span::Edition;
    use stdx::format_to;
    use syntax::ast::AstNode;
    use test_fixture::WithFixture;

    use crate::test_db::TestDB;

    use super::*;

    /// `code` needs to contain a cursor marker; checks that `find_path` for the
    /// item the `path` refers to returns that same path when called from the
    /// module the cursor is in.
    #[track_caller]
    fn check_found_path_(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        path: &str,
        prefer_prelude: bool,
        prefer_absolute: bool,
        prefer_no_std: bool,
        allow_unstable: bool,
        expect: Expect,
    ) {
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

        let (def_map, local_def_map) = module.local_def_map(&db);
        let resolved = def_map
            .resolve_path(
                local_def_map,
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
                ImportPathConfig { prefer_no_std, prefer_prelude, prefer_absolute, allow_unstable },
            );
            format_to!(
                res,
                "{:7}(imports {}): {}\n",
                format!("{:?}", prefix),
                if ignore_local_imports { '✖' } else { '✔' },
                found_path.map_or_else(
                    || "<unresolvable>".to_owned(),
                    |it| it.display(&db, Edition::CURRENT).to_string()
                ),
            );
        }
        expect.assert_eq(&res);
    }

    fn check_found_path(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        path: &str,
        expect: Expect,
    ) {
        check_found_path_(ra_fixture, path, false, false, false, false, expect);
    }

    fn check_found_path_prelude(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        path: &str,
        expect: Expect,
    ) {
        check_found_path_(ra_fixture, path, true, false, false, false, expect);
    }

    fn check_found_path_absolute(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        path: &str,
        expect: Expect,
    ) {
        check_found_path_(ra_fixture, path, false, true, false, false, expect);
    }

    fn check_found_path_prefer_no_std(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        path: &str,
        expect: Expect,
    ) {
        check_found_path_(ra_fixture, path, false, false, true, false, expect);
    }

    fn check_found_path_prefer_no_std_allow_unstable(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        path: &str,
        expect: Expect,
    ) {
        check_found_path_(ra_fixture, path, false, false, true, true, expect);
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
    fn partially_imported_with_prefer_absolute() {
        cov_mark::check!(partially_imported);
        // Similar to partially_imported test case above, but with prefer_absolute enabled.
        // Even if the actual imported item is in external crate, if the path to that item
        // is starting from the imported name, then the path should not start from "::".
        // i.e. The first line in the expected output should not start from "::".
        check_found_path_absolute(
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
                Plain  (imports ✖): ::syntax::ast::ModuleItem
                ByCrate(imports ✔): crate::ast::ModuleItem
                ByCrate(imports ✖): ::syntax::ast::ModuleItem
                BySelf (imports ✔): self::ast::ModuleItem
                BySelf (imports ✖): ::syntax::ast::ModuleItem
            "#]],
        );
    }

    #[test]
    fn same_crate_reexport() {
        check_found_path(
            r#"
mod bar {
    mod foo { pub(crate) struct S; }
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
    mod foo { pub(crate) struct S; }
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
    fn prefer_core_paths_over_std_for_mod_reexport() {
        check_found_path_prefer_no_std(
            r#"
//- /main.rs crate:main deps:core,std

$0

//- /stdlib.rs crate:std deps:core

pub use core::pin;

//- /corelib.rs crate:core

pub mod pin {
    pub struct Pin;
}
            "#,
            "std::pin::Pin",
            expect![[r#"
                Plain  (imports ✔): core::pin::Pin
                Plain  (imports ✖): core::pin::Pin
                ByCrate(imports ✔): core::pin::Pin
                ByCrate(imports ✖): core::pin::Pin
                BySelf (imports ✔): core::pin::Pin
                BySelf (imports ✖): core::pin::Pin
            "#]],
        );
    }

    #[test]
    fn prefer_core_paths_over_std() {
        check_found_path_prefer_no_std(
            r#"
//- /main.rs crate:main deps:core,std

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
    fn from_inside_module2() {
        check_found_path(
            r#"
mod qux {
    pub mod baz {
        pub struct Foo {}
    }

    mod bar {
        fn bar() {
            $0;
        }
    }
}

            "#,
            "crate::qux::baz::Foo",
            expect![[r#"
                Plain  (imports ✔): super::baz::Foo
                Plain  (imports ✖): super::baz::Foo
                ByCrate(imports ✔): crate::qux::baz::Foo
                ByCrate(imports ✖): crate::qux::baz::Foo
                BySelf (imports ✔): super::baz::Foo
                BySelf (imports ✖): super::baz::Foo
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
        check_found_path_prefer_no_std_allow_unstable(
            r#"
//- /main.rs crate:main deps:std,core
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
    fn respects_absolute_setting() {
        let ra_fixture = r#"
//- /main.rs crate:main deps:krate
$0
//- /krate.rs crate:krate
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

        check_found_path_absolute(
            ra_fixture,
            "krate::foo::Foo",
            expect![[r#"
            Plain  (imports ✔): ::krate::foo::Foo
            Plain  (imports ✖): ::krate::foo::Foo
            ByCrate(imports ✔): ::krate::foo::Foo
            ByCrate(imports ✖): ::krate::foo::Foo
            BySelf (imports ✔): ::krate::foo::Foo
            BySelf (imports ✖): ::krate::foo::Foo
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

    #[test]
    fn prefer_long_std_over_short_extern() {
        check_found_path(
            r#"
//- /lib.rs crate:main deps:futures_lite,std,core
$0
//- /futures_lite.rs crate:futures_lite deps:std,core
pub use crate::future::Future;
pub mod future {
    pub use core::future::Future;
}
//- /std.rs crate:std deps:core
pub use core::future;
//- /core.rs crate:core
pub mod future {
    pub trait Future {}
}
"#,
            "core::future::Future",
            expect![[r#"
                Plain  (imports ✔): std::future::Future
                Plain  (imports ✖): std::future::Future
                ByCrate(imports ✔): std::future::Future
                ByCrate(imports ✖): std::future::Future
                BySelf (imports ✔): std::future::Future
                BySelf (imports ✖): std::future::Future
            "#]],
        );
    }
}
