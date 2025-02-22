use std::fmt;
use std::hash::Hash;

use rustc_ast::expand::autodiff_attrs::AutoDiffItem;
use rustc_attr_parsing::InlineAttr;
use rustc_data_structures::base_n::{BaseNString, CASE_INSENSITIVE, ToBaseN};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};
use rustc_data_structures::unord::UnordMap;
use rustc_hashes::Hash128;
use rustc_hir::ItemId;
use rustc_hir::def_id::{CrateNum, DefId, DefIdSet, LOCAL_CRATE};
use rustc_index::Idx;
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use rustc_query_system::ich::StableHashingContext;
use rustc_session::config::OptLevel;
use rustc_span::{Span, Symbol};
use rustc_target::spec::SymbolVisibility;
use tracing::debug;

use crate::dep_graph::{DepNode, WorkProduct, WorkProductId};
use crate::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use crate::ty::{GenericArgs, Instance, InstanceKind, SymbolName, TyCtxt};

/// Describes how a monomorphization will be instantiated in object files.
#[derive(PartialEq)]
pub enum InstantiationMode {
    /// There will be exactly one instance of the given MonoItem. It will have
    /// external linkage so that it can be linked to from other codegen units.
    GloballyShared {
        /// In some compilation scenarios we may decide to take functions that
        /// are typically `LocalCopy` and instead move them to `GloballyShared`
        /// to avoid codegenning them a bunch of times. In this situation,
        /// however, our local copy may conflict with other crates also
        /// inlining the same function.
        ///
        /// This flag indicates that this situation is occurring, and informs
        /// symbol name calculation that some extra mangling is needed to
        /// avoid conflicts. Note that this may eventually go away entirely if
        /// ThinLTO enables us to *always* have a globally shared instance of a
        /// function within one crate's compilation.
        may_conflict: bool,
    },

    /// Each codegen unit containing a reference to the given MonoItem will
    /// have its own private copy of the function (with internal linkage).
    LocalCopy,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash, HashStable, TyEncodable, TyDecodable)]
pub enum MonoItem<'tcx> {
    Fn(Instance<'tcx>),
    Static(DefId),
    GlobalAsm(ItemId),
}

impl<'tcx> MonoItem<'tcx> {
    /// Returns `true` if the mono item is user-defined (i.e. not compiler-generated, like shims).
    pub fn is_user_defined(&self) -> bool {
        match *self {
            MonoItem::Fn(instance) => matches!(instance.def, InstanceKind::Item(..)),
            MonoItem::Static(..) | MonoItem::GlobalAsm(..) => true,
        }
    }

    // Note: if you change how item size estimates work, you might need to
    // change NON_INCR_MIN_CGU_SIZE as well.
    pub fn size_estimate(&self, tcx: TyCtxt<'tcx>) -> usize {
        match *self {
            MonoItem::Fn(instance) => tcx.size_estimate(instance),
            // Conservatively estimate the size of a static declaration or
            // assembly item to be 1.
            MonoItem::Static(_) | MonoItem::GlobalAsm(_) => 1,
        }
    }

    pub fn is_generic_fn(&self) -> bool {
        match self {
            MonoItem::Fn(instance) => instance.args.non_erasable_generics().next().is_some(),
            MonoItem::Static(..) | MonoItem::GlobalAsm(..) => false,
        }
    }

    pub fn symbol_name(&self, tcx: TyCtxt<'tcx>) -> SymbolName<'tcx> {
        match *self {
            MonoItem::Fn(instance) => tcx.symbol_name(instance),
            MonoItem::Static(def_id) => tcx.symbol_name(Instance::mono(tcx, def_id)),
            MonoItem::GlobalAsm(item_id) => {
                SymbolName::new(tcx, &format!("global_asm_{:?}", item_id.owner_id))
            }
        }
    }

    pub fn instantiation_mode(&self, tcx: TyCtxt<'tcx>) -> InstantiationMode {
        // The case handling here is written in the same style as cross_crate_inlinable, we first
        // handle the cases where we must use a particular instantiation mode, then cascade down
        // through a sequence of heuristics.

        // The first thing we do is detect MonoItems which we must instantiate exactly once in the
        // whole program.

        // Statics and global_asm! must be instantiated exactly once.
        let instance = match *self {
            MonoItem::Fn(instance) => instance,
            MonoItem::Static(..) | MonoItem::GlobalAsm(..) => {
                return InstantiationMode::GloballyShared { may_conflict: false };
            }
        };

        // Similarly, the executable entrypoint must be instantiated exactly once.
        if let Some((entry_def_id, _)) = tcx.entry_fn(()) {
            if instance.def_id() == entry_def_id {
                return InstantiationMode::GloballyShared { may_conflict: false };
            }
        }

        // If the function is #[naked] or contains any other attribute that requires exactly-once
        // instantiation:
        let codegen_fn_attrs = tcx.codegen_fn_attrs(instance.def_id());
        if codegen_fn_attrs.contains_extern_indicator()
            || codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED)
        {
            return InstantiationMode::GloballyShared { may_conflict: false };
        }

        // FIXME: The logic for which functions are permitted to get LocalCopy is actually spread
        // across 4 functions:
        // * cross_crate_inlinable(def_id)
        // * InstanceKind::requires_inline
        // * InstanceKind::generate_cgu_internal_copy
        // * MonoItem::instantiation_mode
        // Since reachable_non_generics calls InstanceKind::generates_cgu_internal_copy to decide
        // which symbols this crate exports, we are obligated to only generate LocalCopy when
        // generates_cgu_internal_copy returns true.
        if !instance.def.generates_cgu_internal_copy(tcx) {
            return InstantiationMode::GloballyShared { may_conflict: false };
        }

        // Beginning of heuristics. The handling of link-dead-code and inline(always) are QoL only,
        // the compiler should not crash and linkage should work, but codegen may be undesirable.

        // -Clink-dead-code was given an unfortunate name; the point of the flag is to assist
        // coverage tools which rely on having every function in the program appear in the
        // generated code. If we select LocalCopy, functions which are not used because they are
        // missing test coverage will disappear from such coverage reports, defeating the point.
        // Note that -Cinstrument-coverage does not require such assistance from us, only coverage
        // tools implemented without compiler support ironically require a special compiler flag.
        if tcx.sess.link_dead_code() {
            return InstantiationMode::GloballyShared { may_conflict: true };
        }

        // To ensure that #[inline(always)] can be inlined as much as possible, especially in unoptimized
        // builds, we always select LocalCopy.
        if codegen_fn_attrs.inline.always() {
            return InstantiationMode::LocalCopy;
        }

        // #[inline(never)] functions in general are poor candidates for inlining and thus since
        // LocalCopy generally increases code size for the benefit of optimizations from inlining,
        // we want to give them GloballyShared codegen.
        // The slight problem is that generic functions need to always support cross-crate
        // compilation, so all previous stages of the compiler are obligated to treat generic
        // functions the same as those that unconditionally get LocalCopy codegen. It's only when
        // we get here that we can at least not codegen a #[inline(never)] generic function in all
        // of our CGUs.
        if let InlineAttr::Never = tcx.codegen_fn_attrs(instance.def_id()).inline
            && self.is_generic_fn()
        {
            return InstantiationMode::GloballyShared { may_conflict: true };
        }

        // The fallthrough case is to generate LocalCopy for all optimized builds, and
        // GloballyShared with conflict prevention when optimizations are disabled.
        match tcx.sess.opts.optimize {
            OptLevel::No => InstantiationMode::GloballyShared { may_conflict: true },
            _ => InstantiationMode::LocalCopy,
        }
    }

    pub fn explicit_linkage(&self, tcx: TyCtxt<'tcx>) -> Option<Linkage> {
        let def_id = match *self {
            MonoItem::Fn(ref instance) => instance.def_id(),
            MonoItem::Static(def_id) => def_id,
            MonoItem::GlobalAsm(..) => return None,
        };

        let codegen_fn_attrs = tcx.codegen_fn_attrs(def_id);
        codegen_fn_attrs.linkage
    }

    /// Returns `true` if this instance is instantiable - whether it has no unsatisfied
    /// predicates.
    ///
    /// In order to codegen an item, all of its predicates must hold, because
    /// otherwise the item does not make sense. Type-checking ensures that
    /// the predicates of every item that is *used by* a valid item *do*
    /// hold, so we can rely on that.
    ///
    /// However, we codegen collector roots (reachable items) and functions
    /// in vtables when they are seen, even if they are not used, and so they
    /// might not be instantiable. For example, a programmer can define this
    /// public function:
    ///
    ///     pub fn foo<'a>(s: &'a mut ()) where &'a mut (): Clone {
    ///         <&mut () as Clone>::clone(&s);
    ///     }
    ///
    /// That function can't be codegened, because the method `<&mut () as Clone>::clone`
    /// does not exist. Luckily for us, that function can't ever be used,
    /// because that would require for `&'a mut (): Clone` to hold, so we
    /// can just not emit any code, or even a linker reference for it.
    ///
    /// Similarly, if a vtable method has such a signature, and therefore can't
    /// be used, we can just not emit it and have a placeholder (a null pointer,
    /// which will never be accessed) in its place.
    pub fn is_instantiable(&self, tcx: TyCtxt<'tcx>) -> bool {
        debug!("is_instantiable({:?})", self);
        let (def_id, args) = match *self {
            MonoItem::Fn(ref instance) => (instance.def_id(), instance.args),
            MonoItem::Static(def_id) => (def_id, GenericArgs::empty()),
            // global asm never has predicates
            MonoItem::GlobalAsm(..) => return true,
        };

        !tcx.instantiate_and_check_impossible_predicates((def_id, &args))
    }

    pub fn local_span(&self, tcx: TyCtxt<'tcx>) -> Option<Span> {
        match *self {
            MonoItem::Fn(Instance { def, .. }) => def.def_id().as_local(),
            MonoItem::Static(def_id) => def_id.as_local(),
            MonoItem::GlobalAsm(item_id) => Some(item_id.owner_id.def_id),
        }
        .map(|def_id| tcx.def_span(def_id))
    }

    // Only used by rustc_codegen_cranelift
    pub fn codegen_dep_node(&self, tcx: TyCtxt<'tcx>) -> DepNode {
        crate::dep_graph::make_compile_mono_item(tcx, self)
    }

    /// Returns the item's `CrateNum`
    pub fn krate(&self) -> CrateNum {
        match self {
            MonoItem::Fn(instance) => instance.def_id().krate,
            MonoItem::Static(def_id) => def_id.krate,
            MonoItem::GlobalAsm(..) => LOCAL_CRATE,
        }
    }

    /// Returns the item's `DefId`
    pub fn def_id(&self) -> DefId {
        match *self {
            MonoItem::Fn(Instance { def, .. }) => def.def_id(),
            MonoItem::Static(def_id) => def_id,
            MonoItem::GlobalAsm(item_id) => item_id.owner_id.to_def_id(),
        }
    }
}

impl<'tcx> fmt::Display for MonoItem<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            MonoItem::Fn(instance) => write!(f, "fn {instance}"),
            MonoItem::Static(def_id) => {
                write!(f, "static {}", Instance::new(def_id, GenericArgs::empty()))
            }
            MonoItem::GlobalAsm(..) => write!(f, "global_asm"),
        }
    }
}

impl ToStableHashKey<StableHashingContext<'_>> for MonoItem<'_> {
    type KeyType = Fingerprint;

    fn to_stable_hash_key(&self, hcx: &StableHashingContext<'_>) -> Self::KeyType {
        let mut hasher = StableHasher::new();
        self.hash_stable(&mut hcx.clone(), &mut hasher);
        hasher.finish()
    }
}

#[derive(Debug, HashStable, Copy, Clone)]
pub struct MonoItemPartitions<'tcx> {
    pub codegen_units: &'tcx [CodegenUnit<'tcx>],
    pub all_mono_items: &'tcx DefIdSet,
    pub autodiff_items: &'tcx [AutoDiffItem],
}

#[derive(Debug, HashStable)]
pub struct CodegenUnit<'tcx> {
    /// A name for this CGU. Incremental compilation requires that
    /// name be unique amongst **all** crates. Therefore, it should
    /// contain something unique to this crate (e.g., a module path)
    /// as well as the crate name and disambiguator.
    name: Symbol,
    items: FxIndexMap<MonoItem<'tcx>, MonoItemData>,
    size_estimate: usize,
    primary: bool,
    /// True if this is CGU is used to hold code coverage information for dead code,
    /// false otherwise.
    is_code_coverage_dead_code_cgu: bool,
}

/// Auxiliary info about a `MonoItem`.
#[derive(Copy, Clone, PartialEq, Debug, HashStable)]
pub struct MonoItemData {
    /// A cached copy of the result of `MonoItem::instantiation_mode`, where
    /// `GloballyShared` maps to `false` and `LocalCopy` maps to `true`.
    pub inlined: bool,

    pub linkage: Linkage,
    pub visibility: Visibility,

    /// A cached copy of the result of `MonoItem::size_estimate`.
    pub size_estimate: usize,
}

/// Specifies the linkage type for a `MonoItem`.
///
/// See <https://llvm.org/docs/LangRef.html#linkage-types> for more details about these variants.
#[derive(Copy, Clone, PartialEq, Debug, TyEncodable, TyDecodable, HashStable)]
pub enum Linkage {
    External,
    AvailableExternally,
    LinkOnceAny,
    LinkOnceODR,
    WeakAny,
    WeakODR,
    Internal,
    ExternalWeak,
    Common,
}

/// Specifies the symbol visibility with regards to dynamic linking.
///
/// Visibility doesn't have any effect when linkage is internal.
///
/// DSO means dynamic shared object, that is a dynamically linked executable or dylib.
#[derive(Copy, Clone, PartialEq, Debug, HashStable)]
pub enum Visibility {
    /// Export the symbol from the DSO and apply overrides of the symbol by outside DSOs to within
    /// the DSO if the object file format supports this.
    Default,
    /// Hide the symbol outside of the defining DSO even when external linkage is used to export it
    /// from the object file.
    Hidden,
    /// Export the symbol from the DSO, but don't apply overrides of the symbol by outside DSOs to
    /// within the DSO. Equivalent to default visibility with object file formats that don't support
    /// overriding exported symbols by another DSO.
    Protected,
}

impl From<SymbolVisibility> for Visibility {
    fn from(value: SymbolVisibility) -> Self {
        match value {
            SymbolVisibility::Hidden => Visibility::Hidden,
            SymbolVisibility::Protected => Visibility::Protected,
            SymbolVisibility::Interposable => Visibility::Default,
        }
    }
}

impl<'tcx> CodegenUnit<'tcx> {
    #[inline]
    pub fn new(name: Symbol) -> CodegenUnit<'tcx> {
        CodegenUnit {
            name,
            items: Default::default(),
            size_estimate: 0,
            primary: false,
            is_code_coverage_dead_code_cgu: false,
        }
    }

    pub fn name(&self) -> Symbol {
        self.name
    }

    pub fn set_name(&mut self, name: Symbol) {
        self.name = name;
    }

    pub fn is_primary(&self) -> bool {
        self.primary
    }

    pub fn make_primary(&mut self) {
        self.primary = true;
    }

    pub fn items(&self) -> &FxIndexMap<MonoItem<'tcx>, MonoItemData> {
        &self.items
    }

    pub fn items_mut(&mut self) -> &mut FxIndexMap<MonoItem<'tcx>, MonoItemData> {
        &mut self.items
    }

    pub fn is_code_coverage_dead_code_cgu(&self) -> bool {
        self.is_code_coverage_dead_code_cgu
    }

    /// Marks this CGU as the one used to contain code coverage information for dead code.
    pub fn make_code_coverage_dead_code_cgu(&mut self) {
        self.is_code_coverage_dead_code_cgu = true;
    }

    pub fn mangle_name(human_readable_name: &str) -> BaseNString {
        let mut hasher = StableHasher::new();
        human_readable_name.hash(&mut hasher);
        let hash: Hash128 = hasher.finish();
        hash.as_u128().to_base_fixed_len(CASE_INSENSITIVE)
    }

    pub fn compute_size_estimate(&mut self) {
        // The size of a codegen unit as the sum of the sizes of the items
        // within it.
        self.size_estimate = self.items.values().map(|data| data.size_estimate).sum();
    }

    /// Should only be called if [`compute_size_estimate`] has previously been called.
    ///
    /// [`compute_size_estimate`]: Self::compute_size_estimate
    #[inline]
    pub fn size_estimate(&self) -> usize {
        // Items are never zero-sized, so if we have items the estimate must be
        // non-zero, unless we forgot to call `compute_size_estimate` first.
        assert!(self.items.is_empty() || self.size_estimate != 0);
        self.size_estimate
    }

    pub fn contains_item(&self, item: &MonoItem<'tcx>) -> bool {
        self.items().contains_key(item)
    }

    pub fn work_product_id(&self) -> WorkProductId {
        WorkProductId::from_cgu_name(self.name().as_str())
    }

    pub fn previous_work_product(&self, tcx: TyCtxt<'_>) -> WorkProduct {
        let work_product_id = self.work_product_id();
        tcx.dep_graph
            .previous_work_product(&work_product_id)
            .unwrap_or_else(|| panic!("Could not find work-product for CGU `{}`", self.name()))
    }

    pub fn items_in_deterministic_order(
        &self,
        tcx: TyCtxt<'tcx>,
    ) -> Vec<(MonoItem<'tcx>, MonoItemData)> {
        // The codegen tests rely on items being process in the same order as
        // they appear in the file, so for local items, we sort by node_id first
        #[derive(PartialEq, Eq, PartialOrd, Ord)]
        struct ItemSortKey<'tcx>(Option<usize>, SymbolName<'tcx>);

        fn item_sort_key<'tcx>(tcx: TyCtxt<'tcx>, item: MonoItem<'tcx>) -> ItemSortKey<'tcx> {
            ItemSortKey(
                match item {
                    MonoItem::Fn(ref instance) => {
                        match instance.def {
                            // We only want to take HirIds of user-defined
                            // instances into account. The others don't matter for
                            // the codegen tests and can even make item order
                            // unstable.
                            InstanceKind::Item(def) => def.as_local().map(Idx::index),
                            InstanceKind::VTableShim(..)
                            | InstanceKind::ReifyShim(..)
                            | InstanceKind::Intrinsic(..)
                            | InstanceKind::FnPtrShim(..)
                            | InstanceKind::Virtual(..)
                            | InstanceKind::ClosureOnceShim { .. }
                            | InstanceKind::ConstructCoroutineInClosureShim { .. }
                            | InstanceKind::DropGlue(..)
                            | InstanceKind::CloneShim(..)
                            | InstanceKind::ThreadLocalShim(..)
                            | InstanceKind::FnPtrAddrShim(..)
                            | InstanceKind::AsyncDropGlueCtorShim(..) => None,
                        }
                    }
                    MonoItem::Static(def_id) => def_id.as_local().map(Idx::index),
                    MonoItem::GlobalAsm(item_id) => Some(item_id.owner_id.def_id.index()),
                },
                item.symbol_name(tcx),
            )
        }

        let mut items: Vec<_> = self.items().iter().map(|(&i, &data)| (i, data)).collect();
        items.sort_by_cached_key(|&(i, _)| item_sort_key(tcx, i));
        items
    }

    pub fn codegen_dep_node(&self, tcx: TyCtxt<'tcx>) -> DepNode {
        crate::dep_graph::make_compile_codegen_unit(tcx, self.name())
    }
}

impl ToStableHashKey<StableHashingContext<'_>> for CodegenUnit<'_> {
    type KeyType = String;

    fn to_stable_hash_key(&self, _: &StableHashingContext<'_>) -> Self::KeyType {
        // Codegen unit names are conceptually required to be stable across
        // compilation session so that object file names match up.
        self.name.to_string()
    }
}

pub struct CodegenUnitNameBuilder<'tcx> {
    tcx: TyCtxt<'tcx>,
    cache: UnordMap<CrateNum, String>,
}

impl<'tcx> CodegenUnitNameBuilder<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        CodegenUnitNameBuilder { tcx, cache: Default::default() }
    }

    /// CGU names should fulfill the following requirements:
    /// - They should be able to act as a file name on any kind of file system
    /// - They should not collide with other CGU names, even for different versions
    ///   of the same crate.
    ///
    /// Consequently, we don't use special characters except for '.' and '-' and we
    /// prefix each name with the crate-name and crate-disambiguator.
    ///
    /// This function will build CGU names of the form:
    ///
    /// ```text
    /// <crate-name>.<crate-disambiguator>[-in-<local-crate-id>](-<component>)*[.<special-suffix>]
    /// <local-crate-id> = <local-crate-name>.<local-crate-disambiguator>
    /// ```
    ///
    /// The '.' before `<special-suffix>` makes sure that names with a special
    /// suffix can never collide with a name built out of regular Rust
    /// identifiers (e.g., module paths).
    pub fn build_cgu_name<I, C, S>(
        &mut self,
        cnum: CrateNum,
        components: I,
        special_suffix: Option<S>,
    ) -> Symbol
    where
        I: IntoIterator<Item = C>,
        C: fmt::Display,
        S: fmt::Display,
    {
        let cgu_name = self.build_cgu_name_no_mangle(cnum, components, special_suffix);

        if self.tcx.sess.opts.unstable_opts.human_readable_cgu_names {
            cgu_name
        } else {
            Symbol::intern(&CodegenUnit::mangle_name(cgu_name.as_str()))
        }
    }

    /// Same as `CodegenUnit::build_cgu_name()` but will never mangle the
    /// resulting name.
    pub fn build_cgu_name_no_mangle<I, C, S>(
        &mut self,
        cnum: CrateNum,
        components: I,
        special_suffix: Option<S>,
    ) -> Symbol
    where
        I: IntoIterator<Item = C>,
        C: fmt::Display,
        S: fmt::Display,
    {
        use std::fmt::Write;

        let mut cgu_name = String::with_capacity(64);

        // Start out with the crate name and disambiguator
        let tcx = self.tcx;
        let crate_prefix = self.cache.entry(cnum).or_insert_with(|| {
            // Whenever the cnum is not LOCAL_CRATE we also mix in the
            // local crate's ID. Otherwise there can be collisions between CGUs
            // instantiating stuff for upstream crates.
            let local_crate_id = if cnum != LOCAL_CRATE {
                let local_stable_crate_id = tcx.stable_crate_id(LOCAL_CRATE);
                format!("-in-{}.{:08x}", tcx.crate_name(LOCAL_CRATE), local_stable_crate_id)
            } else {
                String::new()
            };

            let stable_crate_id = tcx.stable_crate_id(LOCAL_CRATE);
            format!("{}.{:08x}{}", tcx.crate_name(cnum), stable_crate_id, local_crate_id)
        });

        write!(cgu_name, "{crate_prefix}").unwrap();

        // Add the components
        for component in components {
            write!(cgu_name, "-{component}").unwrap();
        }

        if let Some(special_suffix) = special_suffix {
            // We add a dot in here so it cannot clash with anything in a regular
            // Rust identifier
            write!(cgu_name, ".{special_suffix}").unwrap();
        }

        Symbol::intern(&cgu_name)
    }
}

/// See module-level docs of `rustc_monomorphize::collector` on some context for "mentioned" items.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable)]
pub enum CollectionMode {
    /// Collect items that are used, i.e., actually needed for codegen.
    ///
    /// Which items are used can depend on optimization levels, as MIR optimizations can remove
    /// uses.
    UsedItems,
    /// Collect items that are mentioned. The goal of this mode is that it is independent of
    /// optimizations: the set of "mentioned" items is computed before optimizations are run.
    ///
    /// The exact contents of this set are *not* a stable guarantee. (For instance, it is currently
    /// computed after drop-elaboration. If we ever do some optimizations even in debug builds, we
    /// might decide to run them before computing mentioned items.) The key property of this set is
    /// that it is optimization-independent.
    MentionedItems,
}
