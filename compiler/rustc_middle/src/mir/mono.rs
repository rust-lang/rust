use crate::dep_graph::{DepNode, WorkProduct, WorkProductId};
use crate::ty::{subst::InternalSubsts, Instance, InstanceDef, SymbolName, TyCtxt};
use rustc_attr::InlineAttr;
use rustc_data_structures::base_n;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{Hash128, HashStable, StableHasher};
use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc_hir::ItemId;
use rustc_index::Idx;
use rustc_query_system::ich::StableHashingContext;
use rustc_session::config::OptLevel;
use rustc_span::source_map::Span;
use rustc_span::symbol::Symbol;
use std::fmt;
use std::hash::Hash;

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

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash, HashStable)]
pub enum MonoItem<'tcx> {
    Fn(Instance<'tcx>),
    Static(DefId),
    GlobalAsm(ItemId),
}

impl<'tcx> MonoItem<'tcx> {
    /// Returns `true` if the mono item is user-defined (i.e. not compiler-generated, like shims).
    pub fn is_user_defined(&self) -> bool {
        match *self {
            MonoItem::Fn(instance) => matches!(instance.def, InstanceDef::Item(..)),
            MonoItem::Static(..) | MonoItem::GlobalAsm(..) => true,
        }
    }

    pub fn size_estimate(&self, tcx: TyCtxt<'tcx>) -> usize {
        match *self {
            MonoItem::Fn(instance) => {
                // Estimate the size of a function based on how many statements
                // it contains.
                tcx.instance_def_size_estimate(instance.def)
            }
            // Conservatively estimate the size of a static declaration
            // or assembly to be 1.
            MonoItem::Static(_) | MonoItem::GlobalAsm(_) => 1,
        }
    }

    pub fn is_generic_fn(&self) -> bool {
        match *self {
            MonoItem::Fn(ref instance) => instance.substs.non_erasable_generics().next().is_some(),
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
        let generate_cgu_internal_copies = tcx
            .sess
            .opts
            .unstable_opts
            .inline_in_all_cgus
            .unwrap_or_else(|| tcx.sess.opts.optimize != OptLevel::No)
            && !tcx.sess.link_dead_code();

        match *self {
            MonoItem::Fn(ref instance) => {
                let entry_def_id = tcx.entry_fn(()).map(|(id, _)| id);
                // If this function isn't inlined or otherwise has an extern
                // indicator, then we'll be creating a globally shared version.
                if tcx.codegen_fn_attrs(instance.def_id()).contains_extern_indicator()
                    || !instance.def.generates_cgu_internal_copy(tcx)
                    || Some(instance.def_id()) == entry_def_id
                {
                    return InstantiationMode::GloballyShared { may_conflict: false };
                }

                // At this point we don't have explicit linkage and we're an
                // inlined function. If we're inlining into all CGUs then we'll
                // be creating a local copy per CGU.
                if generate_cgu_internal_copies {
                    return InstantiationMode::LocalCopy;
                }

                // Finally, if this is `#[inline(always)]` we're sure to respect
                // that with an inline copy per CGU, but otherwise we'll be
                // creating one copy of this `#[inline]` function which may
                // conflict with upstream crates as it could be an exported
                // symbol.
                match tcx.codegen_fn_attrs(instance.def_id()).inline {
                    InlineAttr::Always => InstantiationMode::LocalCopy,
                    _ => InstantiationMode::GloballyShared { may_conflict: true },
                }
            }
            MonoItem::Static(..) | MonoItem::GlobalAsm(..) => {
                InstantiationMode::GloballyShared { may_conflict: false }
            }
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
        let (def_id, substs) = match *self {
            MonoItem::Fn(ref instance) => (instance.def_id(), instance.substs),
            MonoItem::Static(def_id) => (def_id, InternalSubsts::empty()),
            // global asm never has predicates
            MonoItem::GlobalAsm(..) => return true,
        };

        !tcx.subst_and_check_impossible_predicates((def_id, &substs))
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
            MonoItem::Fn(ref instance) => instance.def_id().krate,
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
            MonoItem::Fn(instance) => write!(f, "fn {}", instance),
            MonoItem::Static(def_id) => {
                write!(f, "static {}", Instance::new(def_id, InternalSubsts::empty()))
            }
            MonoItem::GlobalAsm(..) => write!(f, "global_asm"),
        }
    }
}

#[derive(Debug)]
pub struct CodegenUnit<'tcx> {
    /// A name for this CGU. Incremental compilation requires that
    /// name be unique amongst **all** crates. Therefore, it should
    /// contain something unique to this crate (e.g., a module path)
    /// as well as the crate name and disambiguator.
    name: Symbol,
    items: FxHashMap<MonoItem<'tcx>, (Linkage, Visibility)>,
    size_estimate: Option<usize>,
    primary: bool,
    /// True if this is CGU is used to hold code coverage information for dead code,
    /// false otherwise.
    is_code_coverage_dead_code_cgu: bool,
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
    Appending,
    Internal,
    Private,
    ExternalWeak,
    Common,
}

#[derive(Copy, Clone, PartialEq, Debug, HashStable)]
pub enum Visibility {
    Default,
    Hidden,
    Protected,
}

impl<'tcx> CodegenUnit<'tcx> {
    #[inline]
    pub fn new(name: Symbol) -> CodegenUnit<'tcx> {
        CodegenUnit {
            name,
            items: Default::default(),
            size_estimate: None,
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

    /// The order of these items is non-determinstic.
    pub fn items(&self) -> &FxHashMap<MonoItem<'tcx>, (Linkage, Visibility)> {
        &self.items
    }

    /// The order of these items is non-determinstic.
    pub fn items_mut(&mut self) -> &mut FxHashMap<MonoItem<'tcx>, (Linkage, Visibility)> {
        &mut self.items
    }

    pub fn is_code_coverage_dead_code_cgu(&self) -> bool {
        self.is_code_coverage_dead_code_cgu
    }

    /// Marks this CGU as the one used to contain code coverage information for dead code.
    pub fn make_code_coverage_dead_code_cgu(&mut self) {
        self.is_code_coverage_dead_code_cgu = true;
    }

    pub fn mangle_name(human_readable_name: &str) -> String {
        // We generate a 80 bit hash from the name. This should be enough to
        // avoid collisions and is still reasonably short for filenames.
        let mut hasher = StableHasher::new();
        human_readable_name.hash(&mut hasher);
        let hash: Hash128 = hasher.finish();
        let hash = hash.as_u128() & ((1u128 << 80) - 1);
        base_n::encode(hash, base_n::CASE_INSENSITIVE)
    }

    pub fn create_size_estimate(&mut self, tcx: TyCtxt<'tcx>) {
        // Estimate the size of a codegen unit as (approximately) the number of MIR
        // statements it corresponds to.
        self.size_estimate = Some(self.items.keys().map(|mi| mi.size_estimate(tcx)).sum());
    }

    #[inline]
    /// Should only be called if [`create_size_estimate`] has previously been called.
    ///
    /// [`create_size_estimate`]: Self::create_size_estimate
    pub fn size_estimate(&self) -> usize {
        self.size_estimate
            .expect("create_size_estimate must be called before getting a size_estimate")
    }

    pub fn modify_size_estimate(&mut self, delta: usize) {
        *self.size_estimate.as_mut().unwrap() += delta;
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
    ) -> Vec<(MonoItem<'tcx>, (Linkage, Visibility))> {
        // The codegen tests rely on items being process in the same order as
        // they appear in the file, so for local items, we sort by node_id first
        #[derive(PartialEq, Eq, PartialOrd, Ord)]
        pub struct ItemSortKey<'tcx>(Option<usize>, SymbolName<'tcx>);

        fn item_sort_key<'tcx>(tcx: TyCtxt<'tcx>, item: MonoItem<'tcx>) -> ItemSortKey<'tcx> {
            ItemSortKey(
                match item {
                    MonoItem::Fn(ref instance) => {
                        match instance.def {
                            // We only want to take HirIds of user-defined
                            // instances into account. The others don't matter for
                            // the codegen tests and can even make item order
                            // unstable.
                            InstanceDef::Item(def) => def.as_local().map(Idx::index),
                            InstanceDef::VTableShim(..)
                            | InstanceDef::ReifyShim(..)
                            | InstanceDef::Intrinsic(..)
                            | InstanceDef::FnPtrShim(..)
                            | InstanceDef::Virtual(..)
                            | InstanceDef::ClosureOnceShim { .. }
                            | InstanceDef::DropGlue(..)
                            | InstanceDef::CloneShim(..)
                            | InstanceDef::ThreadLocalShim(..)
                            | InstanceDef::FnPtrAddrShim(..) => None,
                        }
                    }
                    MonoItem::Static(def_id) => def_id.as_local().map(Idx::index),
                    MonoItem::GlobalAsm(item_id) => Some(item_id.owner_id.def_id.index()),
                },
                item.symbol_name(tcx),
            )
        }

        let mut items: Vec<_> = self.items().iter().map(|(&i, &l)| (i, l)).collect();
        items.sort_by_cached_key(|&(i, _)| item_sort_key(tcx, i));
        items
    }

    pub fn codegen_dep_node(&self, tcx: TyCtxt<'tcx>) -> DepNode {
        crate::dep_graph::make_compile_codegen_unit(tcx, self.name())
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for CodegenUnit<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let CodegenUnit {
            ref items,
            name,
            // The size estimate is not relevant to the hash
            size_estimate: _,
            primary: _,
            is_code_coverage_dead_code_cgu,
        } = *self;

        name.hash_stable(hcx, hasher);
        is_code_coverage_dead_code_cgu.hash_stable(hcx, hasher);

        let mut items: Vec<(Fingerprint, _)> = items
            .iter()
            .map(|(mono_item, &attrs)| {
                let mut hasher = StableHasher::new();
                mono_item.hash_stable(hcx, &mut hasher);
                let mono_item_fingerprint = hasher.finish();
                (mono_item_fingerprint, attrs)
            })
            .collect();

        items.sort_unstable_by_key(|i| i.0);
        items.hash_stable(hcx, hasher);
    }
}

pub struct CodegenUnitNameBuilder<'tcx> {
    tcx: TyCtxt<'tcx>,
    cache: FxHashMap<CrateNum, String>,
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
                let local_stable_crate_id = tcx.sess.local_stable_crate_id();
                format!("-in-{}.{:08x}", tcx.crate_name(LOCAL_CRATE), local_stable_crate_id)
            } else {
                String::new()
            };

            let stable_crate_id = tcx.sess.local_stable_crate_id();
            format!("{}.{:08x}{}", tcx.crate_name(cnum), stable_crate_id, local_crate_id)
        });

        write!(cgu_name, "{}", crate_prefix).unwrap();

        // Add the components
        for component in components {
            write!(cgu_name, "-{}", component).unwrap();
        }

        if let Some(special_suffix) = special_suffix {
            // We add a dot in here so it cannot clash with anything in a regular
            // Rust identifier
            write!(cgu_name, ".{}", special_suffix).unwrap();
        }

        Symbol::intern(&cgu_name)
    }
}
