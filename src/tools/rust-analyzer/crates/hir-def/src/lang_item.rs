//! Collects lang items: items marked with `#[lang = "..."]` attribute.
//!
//! This attribute to tell the compiler about semi built-in std library
//! features, such as Fn family of traits.
use base_db::SourceDatabase;
use hir_expand::name::Name;
use intern::{Symbol, sym};
use rustc_hash::FxHashMap;
use rustc_type_ir::lang_items::{SolverAdtLangItem, SolverProjectionLangItem, SolverTraitLangItem};
use stdx::impl_from;

use crate::{
    AdtId, AssocItemId, AttrDefId, ConstId, Crate, EnumId, EnumVariantId, FunctionId, ImplId,
    ItemContainerId, MacroId, ModuleDefId, StaticId, StructId, TraitId, TypeAliasId, UnionId,
    attrs::AttrFlags,
    nameres::{DefMap, assoc::TraitItems, crate_def_map, crate_local_def_map},
    unstable_features::UnstableFeatures,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LangItemTarget {
    EnumId(EnumId),
    FunctionId(FunctionId),
    ImplId(ImplId),
    StaticId(StaticId),
    StructId(StructId),
    UnionId(UnionId),
    TypeAliasId(TypeAliasId),
    TraitId(TraitId),
    EnumVariantId(EnumVariantId),
    ConstId(ConstId),
    MacroId(MacroId),
}

impl_from!(
    EnumId, FunctionId, ImplId, StaticId, StructId, UnionId, TypeAliasId, TraitId, EnumVariantId, ConstId, MacroId for LangItemTarget
);

/// Salsa query. This will look for lang items in a specific crate.
#[salsa::tracked(returns(as_deref))]
pub fn crate_lang_items(db: &dyn SourceDatabase, krate: Crate) -> Option<Box<LangItems>> {
    if !UnstableFeatures::query(db, krate).lang_items {
        // First query this, *before* querying the def map, to not create a dependency edge.
        return None;
    }

    let mut lang_items = LangItems::default();

    let _p = tracing::info_span!("crate_lang_items_query", ?krate).entered();

    let crate_def_map = crate_def_map(db, krate);
    for (_, module_data) in crate_def_map.modules() {
        for impl_def in module_data.scope.inherent_impls() {
            lang_items.collect_lang_item(db, impl_def);
            for &(_, assoc) in impl_def.impl_items(db).items.iter() {
                match assoc {
                    AssocItemId::FunctionId(f) => lang_items.collect_lang_item(db, f),
                    AssocItemId::TypeAliasId(t) => lang_items.collect_lang_item(db, t),
                    AssocItemId::ConstId(c) => lang_items.collect_lang_item(db, c),
                }
            }
        }

        for def in module_data.scope.declarations() {
            match def {
                ModuleDefId::TraitId(trait_) => {
                    lang_items.collect_lang_item(db, trait_);
                    TraitItems::query(db, trait_).items.iter().for_each(|&(_, assoc_id)| {
                        match assoc_id {
                            AssocItemId::FunctionId(f) => {
                                lang_items.collect_lang_item(db, f);
                            }
                            AssocItemId::TypeAliasId(alias) => {
                                lang_items.collect_lang_item(db, alias)
                            }
                            AssocItemId::ConstId(c) => lang_items.collect_lang_item(db, c),
                        }
                    });
                }
                ModuleDefId::AdtId(AdtId::EnumId(e)) => {
                    lang_items.collect_lang_item(db, e);
                    e.enum_variants(db).variants.values().for_each(|&(id, _)| {
                        lang_items.collect_lang_item(db, id);
                    });
                }
                ModuleDefId::AdtId(AdtId::StructId(s)) => {
                    lang_items.collect_lang_item(db, s);
                }
                ModuleDefId::AdtId(AdtId::UnionId(u)) => {
                    lang_items.collect_lang_item(db, u);
                }
                ModuleDefId::FunctionId(f) => {
                    lang_items.collect_lang_item(db, f);
                }
                ModuleDefId::StaticId(s) => {
                    lang_items.collect_lang_item(db, s);
                }
                ModuleDefId::TypeAliasId(t) => {
                    lang_items.collect_lang_item(db, t);
                }
                ModuleDefId::ConstId(c) => lang_items.collect_lang_item(db, c),
                _ => {}
            }
        }
    }

    if matches!(krate.data(db).origin, base_db::CrateOrigin::Lang(base_db::LangCrateOrigin::Core)) {
        lang_items.fill_non_lang_core_items(db, crate_def_map);
    }

    lang_items.resolve_manually(db);

    if lang_items.is_empty() {
        None
    } else {
        lang_items.shrink_to_fit();
        Some(Box::new(lang_items))
    }
}

// A firewall query, so that changing crate code for non-`feature(lang_items)` crate will not require recomputing
// the more expensive `lang_items()` query.
#[salsa::tracked(returns(deref))]
fn crate_extern_prelude(db: &dyn SourceDatabase, start_crate: Crate) -> Box<[Crate]> {
    crate_local_def_map(db, start_crate)
        .local(db)
        .extern_prelude()
        .map(|(_, (module, _))| module.krate(db))
        .filter(|&krate| {
            // Some crates declares themselves as extern crate like `extern crate self as core`.
            // Ignore these to prevent cycles.
            krate != start_crate
        })
        .collect()
}

/// Salsa query. Look for a lang items, starting from the specified crate and recursively
/// traversing its dependencies.
#[salsa::tracked(returns(ref))]
pub fn lang_items(db: &dyn SourceDatabase, start_crate: Crate) -> LangItems {
    let _p = tracing::debug_span!("lang_items_query", ?start_crate).entered();

    let mut result = crate_lang_items(db, start_crate).cloned().unwrap_or_default();

    // Our `CrateGraph` eagerly inserts sysroot dependencies like `core` or `std` into dependencies
    // even if the target crate has `#![no_std]`, `#![no_core]` or shadowed sysroot dependencies
    // like `dependencies.std.path = ".."`. So we use `extern_prelude()` instead of
    // `CrateData.dependencies` here, which has already come through such sysroot complexities
    // while nameres.
    //
    // See https://github.com/rust-lang/rust-analyzer/pull/20475 for details.
    for &krate in crate_extern_prelude(db, start_crate) {
        result.merge_prefer_other(lang_items(db, krate));
    }

    result
}

impl LangItems {
    fn collect_lang_item<T>(&mut self, db: &dyn SourceDatabase, item: T)
    where
        T: Into<AttrDefId> + Into<LangItemTarget> + Copy,
    {
        if let Some(lang_item) = AttrFlags::lang_item(db, item.into()) {
            tracing::trace!("found lang item `{lang_item}`");
            self.assign_lang_item(lang_item, item.into());
        }
    }
}

fn resolve_core_trait(
    db: &dyn SourceDatabase,
    core_def_map: &DefMap,
    modules: &[Symbol],
    name: Symbol,
) -> Option<TraitId> {
    let mut current = &core_def_map[core_def_map.root];
    for module in modules {
        let Some((ModuleDefId::ModuleId(cur), _)) =
            current.scope.type_(&Name::new_symbol_root(module.clone()))
        else {
            return None;
        };
        if cur.krate(db) != core_def_map.krate() || cur.block(db) != core_def_map.block_id() {
            return None;
        }
        current = &core_def_map[cur];
    }
    let Some((ModuleDefId::TraitId(trait_), _)) = current.scope.type_(&Name::new_symbol_root(name))
    else {
        return None;
    };
    Some(trait_)
}

fn resolve_core_macro(
    db: &dyn SourceDatabase,
    core_def_map: &DefMap,
    modules: &[Symbol],
    name: Symbol,
) -> Option<MacroId> {
    let mut current = &core_def_map[core_def_map.root];
    for module in modules {
        let Some((ModuleDefId::ModuleId(cur), _)) =
            current.scope.type_(&Name::new_symbol_root(module.clone()))
        else {
            return None;
        };
        if cur.krate(db) != core_def_map.krate() || cur.block(db) != core_def_map.block_id() {
            return None;
        }
        current = &core_def_map[cur];
    }
    current.scope.makro(&Name::new_symbol_root(name))
}

impl LangItems {
    fn resolve_manually(&mut self, db: &dyn SourceDatabase) {
        let parent_trait =
            |lang_item: &mut Option<TraitId>, def: Option<FunctionId>| match def?.loc(db).container
            {
                ItemContainerId::TraitId(trait_) => {
                    *lang_item = Some(trait_);
                    Some(trait_)
                }
                _ => None,
            };
        let assoc_types =
            |trait_: TraitId, assoc_types: &mut [(&mut Option<TypeAliasId>, Symbol)]| {
                let trait_items = trait_.trait_items(db);
                for (assoc_type, name) in assoc_types {
                    **assoc_type =
                        trait_items.associated_type_by_name(&Name::new_symbol_root(name.clone()));
                }
            };
        let methods = |trait_: TraitId, assoc_types: &mut [(&mut Option<FunctionId>, Symbol)]| {
            let trait_items = trait_.trait_items(db);
            for (assoc_type, name) in assoc_types {
                **assoc_type = trait_items.method_by_name(&Name::new_symbol_root(name.clone()));
            }
        };
        (|| {
            let into_future = parent_trait(&mut self.IntoFuture, self.IntoFutureIntoFuture)?;
            assoc_types(into_future, &mut [(&mut self.IntoFutureOutput, sym::Output)]);
            Some(())
        })();

        (|| {
            let into_iterator = parent_trait(&mut self.IntoIterator, self.IntoIterIntoIter)?;
            assoc_types(
                into_iterator,
                &mut [
                    (&mut self.IntoIteratorItem, sym::Item),
                    (&mut self.IntoIterIntoIterType, sym::IntoIter),
                ],
            );
            Some(())
        })();

        (|| {
            assoc_types(self.Iterator?, &mut [(&mut self.IteratorItem, sym::Item)]);
            Some(())
        })();

        (|| {
            assoc_types(self.AsyncIterator?, &mut [(&mut self.AsyncIteratorItem, sym::Item)]);
            Some(())
        })();

        for (op_trait, op_method, op_method_name) in [
            (self.Fn, &mut self.Fn_call, sym::call),
            (self.FnMut, &mut self.FnMut_call_mut, sym::call_mut),
            (self.FnOnce, &mut self.FnOnce_call_once, sym::call_once),
            (self.AsyncFn, &mut self.AsyncFn_async_call, sym::async_call),
            (self.AsyncFnMut, &mut self.AsyncFnMut_async_call_mut, sym::async_call_mut),
            (self.AsyncFnOnce, &mut self.AsyncFnOnce_async_call_once, sym::async_call_once),
            (self.Not, &mut self.Not_not, sym::not),
            (self.Neg, &mut self.Neg_neg, sym::neg),
            (self.Add, &mut self.Add_add, sym::add),
            (self.Mul, &mut self.Mul_mul, sym::mul),
            (self.Sub, &mut self.Sub_sub, sym::sub),
            (self.Div, &mut self.Div_div, sym::div),
            (self.Rem, &mut self.Rem_rem, sym::rem),
            (self.Shl, &mut self.Shl_shl, sym::shl),
            (self.Shr, &mut self.Shr_shr, sym::shr),
            (self.BitXor, &mut self.BitXor_bitxor, sym::bitxor),
            (self.BitOr, &mut self.BitOr_bitor, sym::bitor),
            (self.BitAnd, &mut self.BitAnd_bitand, sym::bitand),
            (self.AddAssign, &mut self.AddAssign_add_assign, sym::add_assign),
            (self.MulAssign, &mut self.MulAssign_mul_assign, sym::mul_assign),
            (self.SubAssign, &mut self.SubAssign_sub_assign, sym::sub_assign),
            (self.DivAssign, &mut self.DivAssign_div_assign, sym::div_assign),
            (self.RemAssign, &mut self.RemAssign_rem_assign, sym::rem_assign),
            (self.ShlAssign, &mut self.ShlAssign_shl_assign, sym::shl_assign),
            (self.ShrAssign, &mut self.ShrAssign_shr_assign, sym::shr_assign),
            (self.BitXorAssign, &mut self.BitXorAssign_bitxor_assign, sym::bitxor_assign),
            (self.BitOrAssign, &mut self.BitOrAssign_bitor_assign, sym::bitor_assign),
            (self.BitAndAssign, &mut self.BitAndAssign_bitand_assign, sym::bitand_assign),
            (self.Debug, &mut self.Debug_fmt, sym::fmt),
            (self.Deref, &mut self.Deref_deref, sym::deref),
            (self.DerefMut, &mut self.DerefMut_deref_mut, sym::deref_mut),
            (self.Index, &mut self.Index_index, sym::index),
            (self.IndexMut, &mut self.IndexMut_index_mut, sym::index_mut),
        ] {
            (|| {
                methods(op_trait?, &mut [(op_method, op_method_name)]);
                Some(())
            })();
        }
        (|| {
            methods(
                self.PartialEq?,
                &mut [(&mut self.PartialEq_eq, sym::eq), (&mut self.PartialEq_ne, sym::ne)],
            );
            Some(())
        })();
        (|| {
            methods(
                self.PartialOrd?,
                &mut [
                    (&mut self.PartialOrd_le, sym::le),
                    (&mut self.PartialOrd_lt, sym::lt),
                    (&mut self.PartialOrd_ge, sym::ge),
                    (&mut self.PartialOrd_gt, sym::gt),
                ],
            );
            Some(())
        })();
        (|| {
            methods(
                self.Drop?,
                &mut [(&mut self.Drop_drop, sym::drop), (&mut self.Drop_pin_drop, sym::pin_drop)],
            );
            Some(())
        })();
    }

    #[inline]
    pub fn get_solver_trait(&self, lang_item: SolverTraitLangItem) -> Option<TraitId> {
        self.solver_trait[lang_item as usize]
    }

    #[inline]
    pub fn get_solver_projection(
        &self,
        lang_item: SolverProjectionLangItem,
    ) -> Option<TypeAliasId> {
        self.solver_projection[lang_item as usize]
    }

    #[inline]
    pub fn get_solver_adt(&self, lang_item: SolverAdtLangItem) -> Option<AdtId> {
        self.solver_adt[lang_item as usize]
    }

    #[inline]
    pub fn get_reverse_solver_trait(&self, trait_: TraitId) -> Option<SolverTraitLangItem> {
        self.reverse_solver_trait.get(&trait_).copied()
    }

    #[inline]
    pub fn get_reverse_solver_adt(&self, adt: AdtId) -> Option<SolverAdtLangItem> {
        self.reverse_solver_adt.get(&adt).copied()
    }

    #[inline]
    pub fn get_reverse_solver_projection(
        &self,
        projection: TypeAliasId,
    ) -> Option<SolverProjectionLangItem> {
        self.reverse_solver_projection.get(&projection).copied()
    }

    fn shrink_to_fit(&mut self) {
        let Self { reverse_solver_adt, reverse_solver_projection, reverse_solver_trait, .. } = self;
        reverse_solver_adt.shrink_to_fit();
        reverse_solver_projection.shrink_to_fit();
        reverse_solver_trait.shrink_to_fit();
    }

    fn assign_solver_trait(&mut self, lang_item: SolverTraitLangItem, def: TraitId) {
        self.solver_trait[lang_item as usize] = Some(def);
        self.reverse_solver_trait.insert(def, lang_item);
    }

    fn assign_solver_projection(&mut self, lang_item: SolverProjectionLangItem, def: TypeAliasId) {
        self.solver_projection[lang_item as usize] = Some(def);
        self.reverse_solver_projection.insert(def, lang_item);
    }

    fn assign_solver_adt(&mut self, lang_item: SolverAdtLangItem, def: impl Into<AdtId>) {
        let def = def.into();
        self.solver_adt[lang_item as usize] = Some(def);
        self.reverse_solver_adt.insert(def, lang_item);
    }
}

#[salsa::tracked(returns(as_deref))]
pub fn crate_notable_traits(db: &dyn SourceDatabase, krate: Crate) -> Option<Box<[TraitId]>> {
    let mut traits = Vec::new();

    let crate_def_map = crate_def_map(db, krate);

    for (_, module_data) in crate_def_map.modules() {
        for def in module_data.scope.declarations() {
            if let ModuleDefId::TraitId(trait_) = def
                && AttrFlags::query(db, trait_.into()).contains(AttrFlags::IS_DOC_NOTABLE_TRAIT)
            {
                traits.push(trait_);
            }
        }
    }

    if traits.is_empty() { None } else { Some(traits.into_iter().collect()) }
}

macro_rules! assign_solver {
    ( $self:ident, $def:ident, $lang_item:ident, TraitId, solver ) => {
        $self.assign_solver_trait(SolverTraitLangItem::$lang_item, $def);
    };
    ( $self:ident, $def:ident, $lang_item:ident, TypeAliasId, solver ) => {
        $self.assign_solver_projection(SolverProjectionLangItem::$lang_item, $def);
    };
    ( $self:ident, $def:ident, $lang_item:ident, StructId, solver ) => {
        $self.assign_solver_adt(SolverAdtLangItem::$lang_item, $def);
    };
    ( $self:ident, $def:ident, $lang_item:ident, EnumId, solver ) => {
        $self.assign_solver_adt(SolverAdtLangItem::$lang_item, $def);
    };
    ( $self:ident, $def:ident, $lang_item:ident, UnionId, solver ) => {
        $self.assign_solver_adt(SolverAdtLangItem::$lang_item, $def);
    };
    ( $self:ident, $def:ident, $lang_item:ident, $ty:ident ) => {};
}

macro_rules! solver_exhaustiveness_check {
    (
        $dollar:tt, $solver_ty:ident, $id_ty:ident,
        [ $($id_tys:ident),* ]

        $( $lang_item:ident, $target:ident $(, $solver:ident)?; )*
    ) => {
        const _: () = {
            #[expect(unreachable_patterns, dead_code)]
            fn exhaustiveness_check(lang_item: $solver_ty) {
                macro_rules! one_variant {
                    $(
                        (
                            $dollar lang_item:ident, $id_tys, solver
                        ) => {
                            ($solver_ty::$dollar lang_item, true)
                        };
                    )*
                    ( $dollar($t:tt)* ) => {
                        (_, false)
                    };
                }

                match (lang_item, true) {
                    $(
                        one_variant!( $lang_item, $target $(, $solver )? ) => {}
                    )*
                }
            }
        };
    };
}

macro_rules! get_solver_count {
    (
        $dollar:tt, $solver_ty:ident,
        [ $($id_tys:ident),* ]

        $( $lang_item:ident, $target:ident $(, $solver:ident)?; )*
    ) => {{
        let mut max_discriminant = 0;
        macro_rules! one_variant {
            $(
                (
                    $dollar lang_item:ident, $id_tys, solver
                ) => {
                    if $solver_ty::$dollar lang_item as usize > max_discriminant {
                        max_discriminant = $solver_ty::$dollar lang_item as usize;
                    }
                };
            )*
            ( $dollar($t:tt)* ) => {};
        }
        $(
            one_variant!($lang_item, $target $(, $solver)?);
        )*
        max_discriminant + 1
    }};
}

fn merge_array_prefer_other<T: Copy, const N: usize>(
    this: &mut [Option<T>; N],
    other: &[Option<T>; N],
) {
    for (this, other) in std::iter::zip(this, other) {
        *this = other.or(*this);
    }
}

macro_rules! language_item_table {
    (
        $dollar:tt
        $LangItems:ident =>
        $( $lang_item:ident, $module:ident :: $name:ident, $target:ident $(, $solver:ident)?; )*

        @non_lang_core_traits:

        $( core::$($non_lang_trait_module:ident)::*, $non_lang_trait:ident; )*

        @non_lang_core_macros:

        $( core::$($non_lang_macro_module:ident)::*, $non_lang_macro:ident, $non_lang_macro_field:ident; )*

        @resolve_manually:

        $( $resolve_manually:ident, $resolve_manually_type:ident; )*
    ) => {
        const SOLVER_TRAITS_LEN: usize = get_solver_count! {
            $dollar, SolverTraitLangItem,
            [ TraitId ]

            $( $lang_item, $target $(, $solver)?; )*
        };
        const SOLVER_PROJECTIONS_LEN: usize = get_solver_count! {
            $dollar, SolverProjectionLangItem,
            [ TypeAliasId ]

            $( $lang_item, $target $(, $solver)?; )*
        };
        const SOLVER_ADTS_LEN: usize = get_solver_count! {
            $dollar, SolverAdtLangItem,
            [ StructId, EnumId, UnionId ]

            $( $lang_item, $target $(, $solver)?; )*
        };

        #[allow(non_snake_case)] // FIXME: Should we remove this?
        #[derive(Default, Clone)]
        pub struct $LangItems {
            $(
                pub $lang_item: Option<$target>,
            )*
            $(
                pub $non_lang_trait: Option<TraitId>,
            )*
            $(
                pub $non_lang_macro_field: Option<MacroId>,
            )*
            $(
                pub $resolve_manually: Option<$resolve_manually_type>,
            )*

            solver_trait: [Option<TraitId>; SOLVER_TRAITS_LEN],
            solver_adt: [Option<AdtId>; SOLVER_ADTS_LEN],
            solver_projection: [Option<TypeAliasId>; SOLVER_PROJECTIONS_LEN],

            reverse_solver_trait: FxHashMap<TraitId, SolverTraitLangItem>,
            reverse_solver_adt: FxHashMap<AdtId, SolverAdtLangItem>,
            reverse_solver_projection: FxHashMap<TypeAliasId, SolverProjectionLangItem>,
        }

        impl std::fmt::Debug for $LangItems {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut debug = f.debug_struct(stringify!($LangItems));
                let Self {
                    $( $lang_item, )*
                    $( $non_lang_trait, )*
                    $( $non_lang_macro_field, )*
                    $( $resolve_manually, )*

                    solver_trait: _,
                    solver_adt: _,
                    solver_projection: _,
                    reverse_solver_trait: _,
                    reverse_solver_adt: _,
                    reverse_solver_projection: _,
                } = self;
                $(
                    if let Some(it) = $lang_item {
                        debug.field(stringify!($lang_item), it);
                    }
                )*
                $(
                    if let Some(it) = $non_lang_trait {
                        debug.field(stringify!($non_lang_trait), it);
                    }
                )*
                $(
                    if let Some(it) = $non_lang_macro_field {
                        debug.field(stringify!($non_lang_macro_field), it);
                    }
                )*
                $(
                    if let Some(it) = $resolve_manually {
                        debug.field(stringify!($resolve_manually), it);
                    }
                )*
                debug.finish()
            }
        }

        impl PartialEq for $LangItems {
            fn eq(&self, other: &Self) -> bool {
                let Self {
                    $( $lang_item, )*
                    $( $non_lang_trait, )*
                    $( $non_lang_macro_field, )*
                    $( $resolve_manually, )*

                    solver_trait: _,
                    solver_adt: _,
                    solver_projection: _,
                    reverse_solver_trait: _,
                    reverse_solver_adt: _,
                    reverse_solver_projection: _,
                } = *self;

                true
                    $( && $lang_item == other.$lang_item )*
                    $( && $non_lang_trait == other.$non_lang_trait )*
                    $( && $non_lang_macro_field == other.$non_lang_macro_field )*
                    $( && $resolve_manually == other.$resolve_manually )*
            }
        }

        impl Eq for $LangItems {}

        impl LangItems {
            fn is_empty(&self) -> bool {
                let Self {
                    $( $lang_item, )*
                    $( $non_lang_trait, )*
                    $( $non_lang_macro_field, )*
                    $( $resolve_manually, )*

                    solver_trait: _,
                    solver_adt: _,
                    solver_projection: _,
                    reverse_solver_trait: _,
                    reverse_solver_adt: _,
                    reverse_solver_projection: _,
                } = *self;

                true
                    $( && $lang_item.is_none() )*
                    $( && $non_lang_trait.is_none() )*
                    $( && $non_lang_macro_field.is_none() )*
                    $( && $resolve_manually.is_none() )*
            }

            /// Merges `self` with `other`, with preference to `other` items.
            fn merge_prefer_other(&mut self, other: &Self) {
                $( self.$lang_item = other.$lang_item.or(self.$lang_item); )*
                $( self.$non_lang_trait = other.$non_lang_trait.or(self.$non_lang_trait); )*
                $( self.$non_lang_macro_field = other.$non_lang_macro_field.or(self.$non_lang_macro_field); )*
                $( self.$resolve_manually = other.$resolve_manually.or(self.$resolve_manually); )*

                merge_array_prefer_other(&mut self.solver_trait, &other.solver_trait);
                merge_array_prefer_other(&mut self.solver_adt, &other.solver_adt);
                merge_array_prefer_other(&mut self.solver_projection, &other.solver_projection);

                self.reverse_solver_trait.extend(&other.reverse_solver_trait);
                self.reverse_solver_adt.extend(&other.reverse_solver_adt);
                self.reverse_solver_projection.extend(&other.reverse_solver_projection);
            }

            fn assign_lang_item(&mut self, name: Symbol, target: LangItemTarget) {
                match name {
                    $(
                        _ if name == $module::$name => {
                            if let LangItemTarget::$target(target) = target {
                                self.$lang_item = Some(target);

                                assign_solver!(self, target, $lang_item, $target $(, $solver)? );
                            }
                        }
                    )*
                    _ => {}
                }
            }

            fn fill_non_lang_core_items(&mut self, db: &dyn SourceDatabase, core_def_map: &DefMap) {
                $( self.$non_lang_trait = resolve_core_trait(db, core_def_map, &[ $(sym::$non_lang_trait_module),* ], sym::$non_lang_trait); )*
                $( self.$non_lang_macro_field = resolve_core_macro(db, core_def_map, &[ $(sym::$non_lang_macro_module),* ], sym::$non_lang_macro); )*
            }
        }

        solver_exhaustiveness_check! {
            $dollar, SolverTraitLangItem, TraitId,
            [ TraitId ]

            $( $lang_item, $target $(, $solver)?; )*
        }

        solver_exhaustiveness_check! {
            $dollar, SolverProjectionLangItem, TypeAliasId,
            [ TypeAliasId ]

            $( $lang_item, $target $(, $solver)?; )*
        }

        solver_exhaustiveness_check! {
            $dollar, SolverAdtLangItem, AdtId,
            [ StructId, EnumId, UnionId ]

            $( $lang_item, $target $(, $solver)?; )*
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[allow(non_camel_case_types)]
        pub enum LangItemEnum {
            $(
                $lang_item,
            )*
            $( $non_lang_trait, )*
            $( $non_lang_macro_field, )*
            $( $resolve_manually, )*
        }

        impl LangItemEnum {
            #[inline]
            pub fn from_lang_items(self, lang_items: &LangItems) -> Option<LangItemTarget> {
                match self {
                    $( LangItemEnum::$lang_item => lang_items.$lang_item.map(Into::into), )*
                    $( LangItemEnum::$non_lang_trait => lang_items.$non_lang_trait.map(Into::into), )*
                    $( LangItemEnum::$non_lang_macro_field => lang_items.$non_lang_macro_field.map(Into::into), )*
                    $( LangItemEnum::$resolve_manually => lang_items.$resolve_manually.map(Into::into), )*
                }
            }

            #[inline]
            pub fn from_symbol(symbol: &Symbol) -> Option<Self> {
                match symbol {
                    $( _ if *symbol == $module::$name => Some(Self::$lang_item), )*
                    _ => None,
                }
            }
        }
    }
}

language_item_table! { $ LangItems =>
//  Variant name,            Name,                     Target;
    Sized,                   sym::sized,               TraitId,               solver;
    MetaSized,               sym::meta_sized,          TraitId,               solver;
    PointeeSized,            sym::pointee_sized,       TraitId,               solver;
    Unsize,                  sym::unsize,              TraitId,               solver;
    // Trait injected by `#[derive(PartialEq)]`, (i.e. "Partial EQ").
    StructuralPeq,           sym::structural_peq,      TraitId;
    // Trait injected by `#[derive(Eq)]`, (i.e. "Total EQ"; no, I will not apologize).
    StructuralTeq,           sym::structural_teq,      TraitId;
    Copy,                    sym::copy,                TraitId,               solver;
    UseCloned,               sym::use_cloned,          TraitId;
    Clone,                   sym::clone,               TraitId,               solver;
    TrivialClone,            sym::trivial_clone,       TraitId,               solver;
    Sync,                    sym::sync,                TraitId;
    DiscriminantKind,        sym::discriminant_kind,   TraitId,               solver;
    // The associated item of the `DiscriminantKind` trait.
    Discriminant,            sym::discriminant_type,   TypeAliasId;

    PointeeTrait,            sym::pointee_trait,       TraitId,               solver;
    Metadata,                sym::metadata_type,       TypeAliasId,           solver;
    DynMetadata,             sym::dyn_metadata,        StructId,              solver;

    Freeze,                  sym::freeze,              TraitId;

    NonNull,                 sym::non_null,            StructId;

    FnPtrTrait,              sym::fn_ptr_trait,        TraitId,               solver;
    FnPtrAsPtr,              sym::fn_ptr_as_ptr,       FunctionId;
    FnPtrFromPtr,            sym::fn_ptr_from_ptr,     FunctionId;

    Drop,                    sym::drop,                TraitId,               solver;
    Destruct,                sym::destruct,            TraitId,               solver;
    BikeshedGuaranteedNoDrop,sym::bikeshed_guaranteed_no_drop, TraitId,       solver;

    CoerceUnsized,           sym::coerce_unsized,      TraitId;
    DispatchFromDyn,         sym::dispatch_from_dyn,   TraitId;
    Reborrow,                sym::reborrow,            TraitId;
    CoerceShared,            sym::coerce_shared,       TraitId;

    // language items relating to transmutability
    TransmuteOpts,           sym::transmute_opts,      StructId;
    TransmuteTrait,          sym::transmute_trait,     TraitId,               solver;

    Add,                     sym::add,                 TraitId;
    Sub,                     sym::sub,                 TraitId;
    Mul,                     sym::mul,                 TraitId;
    Div,                     sym::div,                 TraitId;
    Rem,                     sym::rem,                 TraitId;
    Neg,                     sym::neg,                 TraitId;
    Not,                     sym::not,                 TraitId;
    BitXor,                  sym::bitxor,              TraitId;
    BitAnd,                  sym::bitand,              TraitId;
    BitOr,                   sym::bitor,               TraitId;
    Shl,                     sym::shl,                 TraitId;
    Shr,                     sym::shr,                 TraitId;
    AddAssign,               sym::add_assign,          TraitId;
    SubAssign,               sym::sub_assign,          TraitId;
    MulAssign,               sym::mul_assign,          TraitId;
    DivAssign,               sym::div_assign,          TraitId;
    RemAssign,               sym::rem_assign,          TraitId;
    BitXorAssign,            sym::bitxor_assign,       TraitId;
    BitAndAssign,            sym::bitand_assign,       TraitId;
    BitOrAssign,             sym::bitor_assign,        TraitId;
    ShlAssign,               sym::shl_assign,          TraitId;
    ShrAssign,               sym::shr_assign,          TraitId;
    Index,                   sym::index,               TraitId;
    IndexMut,                sym::index_mut,           TraitId;

    UnsafeCell,              sym::unsafe_cell,         StructId;
    UnsafePinned,            sym::unsafe_pinned,       StructId;
    VaList,                  sym::va_list,             StructId;

    Deref,                   sym::deref,               TraitId;
    DerefMut,                sym::deref_mut,           TraitId;
    DerefPure,               sym::deref_pure,          TraitId;
    DerefTarget,             sym::deref_target,        TypeAliasId;
    Receiver,                sym::receiver,            TraitId;
    ReceiverTarget,          sym::receiver_target,     TypeAliasId;

    Fn,                      sym::fn_,                 TraitId,                solver;
    FnMut,                   sym::fn_mut,              TraitId,                solver;
    FnOnce,                  sym::fn_once,             TraitId,                solver;
    AsyncFn,                 sym::async_fn,            TraitId,                solver;
    AsyncFnMut,              sym::async_fn_mut,        TraitId,                solver;
    AsyncFnOnce,             sym::async_fn_once,       TraitId,                solver;
    AsyncFnKindHelper,       sym::async_fn_kind_helper,TraitId,                solver;
    AsyncFnKindUpvars,       sym::async_fn_kind_upvars,TypeAliasId,            solver;

    CallRefFuture,           sym::call_ref_future,     TypeAliasId,            solver;
    CallOnceFuture,          sym::call_once_future,    TypeAliasId,            solver;
    AsyncFnOnceOutput,       sym::async_fn_once_output, TypeAliasId,           solver;

    FnOnceOutput,            sym::fn_once_output,      TypeAliasId;

    Future,                  sym::future_trait,        TraitId,                solver;
    AsyncIterator,           sym::async_iterator,      TraitId,                solver;
    CoroutineState,          sym::coroutine_state,     EnumId;
    Coroutine,               sym::coroutine,           TraitId,                solver;
    CoroutineReturn,         sym::coroutine_return,    TypeAliasId,            solver;
    CoroutineYield,          sym::coroutine_yield,     TypeAliasId,            solver;
    Unpin,                   sym::unpin,               TraitId,                solver;
    Pin,                     sym::pin,                 StructId;

    PartialEq,               sym::eq,                  TraitId;
    PartialOrd,              sym::partial_ord,         TraitId;
    CVoid,                   sym::c_void,              EnumId;

    // A number of panic-related lang items. The `panic` item corresponds to divide-by-zero and
    // various panic cases with `match`. The `panic_bounds_check` item is for indexing arrays.
    //
    // The `begin_unwind` lang item has a predefined symbol name and is sort of a "weak lang item"
    // in the sense that a crate is not required to have it defined to use it, but a final product
    // is required to define it somewhere. Additionally, there are restrictions on crates that use
    // a weak lang item, but do not have it defined.
    Panic,                   sym::panic,               FunctionId;
    PanicNounwind,           sym::panic_nounwind,      FunctionId;
    PanicFmt,                sym::panic_fmt,           FunctionId;
    PanicDisplay,            sym::panic_display,       FunctionId;
    ConstPanicFmt,           sym::const_panic_fmt,     FunctionId;
    PanicBoundsCheck,        sym::panic_bounds_check,  FunctionId;
    PanicMisalignedPointerDereference, sym::panic_misaligned_pointer_dereference, FunctionId;
    PanicInfo,               sym::panic_info,          StructId;
    PanicLocation,           sym::panic_location,      StructId;
    PanicImpl,               sym::panic_impl,          FunctionId;
    PanicCannotUnwind,       sym::panic_cannot_unwind, FunctionId;
    PanicNullPointerDereference, sym::panic_null_pointer_dereference, FunctionId;
    // libstd panic entry point. Necessary for const eval to be able to catch it
    BeginPanic,              sym::begin_panic,         FunctionId;

    // Lang items needed for `format_args!()`.
    FormatAlignment,         sym::format_alignment,    EnumId;
    FormatArgument,          sym::format_argument,     StructId;
    FormatArguments,         sym::format_arguments,    StructId;
    FormatCount,             sym::format_count,        EnumId;
    FormatPlaceholder,       sym::format_placeholder,  StructId;
    FormatUnsafeArg,         sym::format_unsafe_arg,   StructId;

    ExchangeMalloc,          sym::exchange_malloc,     FunctionId;
    BoxFree,                 sym::box_free,            FunctionId;
    DropInPlace,             sym::drop_in_place,       FunctionId;
    AllocLayout,             sym::alloc_layout,        StructId;

    Start,                   sym::start,               FunctionId;

    EhPersonality,           sym::eh_personality,      FunctionId;
    EhCatchTypeinfo,         sym::eh_catch_typeinfo,   StaticId;

    OwnedBox,                sym::owned_box,           StructId;

    PhantomData,             sym::phantom_data,        StructId;

    ManuallyDrop,            sym::manually_drop,       StructId;

    MaybeUninit,             sym::maybe_uninit,        UnionId;

    // Align offset for stride != 1; must not panic.
    AlignOffset,             sym::align_offset,        FunctionId;

    Termination,             sym::termination,         TraitId;

    Try,                     sym::Try,                 TraitId;

    Tuple,                   sym::tuple_trait,         TraitId,               solver;

    SliceLen,                sym::slice_len_fn,        FunctionId;

    // Language items from AST lowering
    TryTraitFromResidual,    sym::from_residual,       FunctionId;
    TryTraitFromOutput,      sym::from_output,         FunctionId;
    TryTraitBranch,          sym::branch,              FunctionId;
    TryTraitFromYeet,        sym::from_yeet,           FunctionId;
    ResidualIntoTryType,     sym::into_try_type,       FunctionId;

    PointerLike,             sym::pointer_like,        TraitId;

    ConstParamTy,            sym::const_param_ty,      TraitId;

    Poll,                    sym::Poll,                EnumId,                solver;
    PollReady,               sym::Ready,               EnumVariantId;
    PollPending,             sym::Pending,             EnumVariantId;

    // FIXME(swatinem): the following lang items are used for async lowering and
    // should become obsolete eventually.
    ResumeTy,                sym::ResumeTy,            StructId;
    GetContext,              sym::get_context,         FunctionId;

    Context,                 sym::Context,             StructId;
    FuturePoll,              sym::poll,                FunctionId;
    FutureOutput,            sym::future_output,       TypeAliasId,           solver;

    Option,                  sym::Option,              EnumId,                solver;
    OptionSome,              sym::Some,                EnumVariantId;
    OptionNone,              sym::None,                EnumVariantId;

    ResultOk,                sym::Ok,                  EnumVariantId;
    ResultErr,               sym::Err,                 EnumVariantId;

    ControlFlowContinue,     sym::Continue,            EnumVariantId;
    ControlFlowBreak,        sym::Break,               EnumVariantId;

    IntoFutureIntoFuture,    sym::into_future,         FunctionId;
    IntoIterIntoIter,        sym::into_iter,           FunctionId;
    IteratorNext,            sym::next,                FunctionId;
    Iterator,                sym::iterator,            TraitId,               solver;
    FusedIterator,           sym::fused_iterator,      TraitId,               solver;

    PinNewUnchecked,         sym::new_unchecked,       FunctionId;

    RangeFrom,               sym::RangeFrom,           StructId;
    RangeFull,               sym::RangeFull,           StructId;
    RangeInclusiveStruct,    sym::RangeInclusive,      StructId;
    RangeInclusiveNew,       sym::range_inclusive_new, FunctionId;
    Range,                   sym::Range,               StructId;
    RangeToInclusive,        sym::RangeToInclusive,    StructId;
    RangeTo,                 sym::RangeTo,             StructId;

    RangeFromCopy,           sym::RangeFromCopy,           StructId;
    RangeInclusiveCopy,      sym::RangeInclusiveCopy,      StructId;
    RangeCopy,               sym::RangeCopy,               StructId;
    RangeToInclusiveCopy,    sym::RangeToInclusiveCopy,    StructId;

    String,                  sym::String,              StructId;
    CStr,                    sym::CStr,                StructId;
    Ordering,                sym::Ordering,            EnumId;

    Field,                   sym::field,               TraitId,               solver;
    FieldBase,               sym::field_base,          TypeAliasId,           solver;
    FieldType,               sym::field_type,          TypeAliasId,           solver;

    RangeMin,                sym::RangeMin,            ConstId;
    RangeMax,                sym::RangeMax,            ConstId;
    RangeSub,                sym::RangeSub,            FunctionId;

    @non_lang_core_traits:
    core::default, Default;
    core::fmt, Debug;
    core::hash, Hash;
    core::cmp, Ord;
    core::cmp, Eq;

    @non_lang_core_macros:
    core::default, Default, DefaultDerive;
    core::fmt, Debug, DebugDerive;
    core::hash, Hash, HashDerive;
    core::cmp, PartialOrd, PartialOrdDerive;
    core::cmp, Ord, OrdDerive;
    core::cmp, PartialEq, PartialEqDerive;
    core::cmp, Eq, EqDerive;
    core::marker, CoercePointee, CoercePointeeDerive;
    core::marker, CoerceShared, CoerceSharedDerive;
    core::marker, Copy, CopyDerive;
    core::marker, Reborrow, ReborrowDerive;
    core::clone, Clone, CloneDerive;

    @resolve_manually:

    IntoFuture,                    TraitId;
    IntoFutureOutput,              TypeAliasId;
    IntoIterator,                  TraitId;
    IntoIteratorItem,              TypeAliasId;
    IntoIterIntoIterType,          TypeAliasId;
    IteratorItem,                  TypeAliasId;
    AsyncIteratorItem,             TypeAliasId;

    Fn_call,                       FunctionId;
    FnMut_call_mut,                FunctionId;
    FnOnce_call_once,              FunctionId;
    AsyncFn_async_call,            FunctionId;
    AsyncFnMut_async_call_mut,     FunctionId;
    AsyncFnOnce_async_call_once,   FunctionId;
    Not_not,                       FunctionId;
    Neg_neg,                       FunctionId;
    Add_add,                       FunctionId;
    Mul_mul,                       FunctionId;
    Sub_sub,                       FunctionId;
    Div_div,                       FunctionId;
    Rem_rem,                       FunctionId;
    Shl_shl,                       FunctionId;
    Shr_shr,                       FunctionId;
    BitXor_bitxor,                 FunctionId;
    BitOr_bitor,                   FunctionId;
    BitAnd_bitand,                 FunctionId;
    AddAssign_add_assign,          FunctionId;
    MulAssign_mul_assign,          FunctionId;
    SubAssign_sub_assign,          FunctionId;
    DivAssign_div_assign,          FunctionId;
    RemAssign_rem_assign,          FunctionId;
    ShlAssign_shl_assign,          FunctionId;
    ShrAssign_shr_assign,          FunctionId;
    BitXorAssign_bitxor_assign,    FunctionId;
    BitOrAssign_bitor_assign,      FunctionId;
    BitAndAssign_bitand_assign,    FunctionId;
    PartialEq_eq,                  FunctionId;
    PartialEq_ne,                  FunctionId;
    PartialOrd_le,                 FunctionId;
    PartialOrd_lt,                 FunctionId;
    PartialOrd_ge,                 FunctionId;
    PartialOrd_gt,                 FunctionId;
    Drop_drop,                     FunctionId;
    Drop_pin_drop,                     FunctionId;
    Debug_fmt,                     FunctionId;
    Deref_deref,                   FunctionId;
    DerefMut_deref_mut,            FunctionId;
    Index_index,                   FunctionId;
    IndexMut_index_mut,            FunctionId;
}
