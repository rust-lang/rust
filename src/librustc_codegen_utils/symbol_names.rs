//! The Rust Linkage Model and Symbol Names
//! =======================================
//!
//! The semantic model of Rust linkage is, broadly, that "there's no global
//! namespace" between crates. Our aim is to preserve the illusion of this
//! model despite the fact that it's not *quite* possible to implement on
//! modern linkers. We initially didn't use system linkers at all, but have
//! been convinced of their utility.
//!
//! There are a few issues to handle:
//!
//!  - Linkers operate on a flat namespace, so we have to flatten names.
//!    We do this using the C++ namespace-mangling technique. Foo::bar
//!    symbols and such.
//!
//!  - Symbols for distinct items with the same *name* need to get different
//!    linkage-names. Examples of this are monomorphizations of functions or
//!    items within anonymous scopes that end up having the same path.
//!
//!  - Symbols in different crates but with same names "within" the crate need
//!    to get different linkage-names.
//!
//!  - Symbol names should be deterministic: Two consecutive runs of the
//!    compiler over the same code base should produce the same symbol names for
//!    the same items.
//!
//!  - Symbol names should not depend on any global properties of the code base,
//!    so that small modifications to the code base do not result in all symbols
//!    changing. In previous versions of the compiler, symbol names incorporated
//!    the SVH (Stable Version Hash) of the crate. This scheme turned out to be
//!    infeasible when used in conjunction with incremental compilation because
//!    small code changes would invalidate all symbols generated previously.
//!
//!  - Even symbols from different versions of the same crate should be able to
//!    live next to each other without conflict.
//!
//! In order to fulfill the above requirements the following scheme is used by
//! the compiler:
//!
//! The main tool for avoiding naming conflicts is the incorporation of a 64-bit
//! hash value into every exported symbol name. Anything that makes a difference
//! to the symbol being named, but does not show up in the regular path needs to
//! be fed into this hash:
//!
//! - Different monomorphizations of the same item have the same path but differ
//!   in their concrete type parameters, so these parameters are part of the
//!   data being digested for the symbol hash.
//!
//! - Rust allows items to be defined in anonymous scopes, such as in
//!   `fn foo() { { fn bar() {} } { fn bar() {} } }`. Both `bar` functions have
//!   the path `foo::bar`, since the anonymous scopes do not contribute to the
//!   path of an item. The compiler already handles this case via so-called
//!   disambiguating `DefPaths` which use indices to distinguish items with the
//!   same name. The DefPaths of the functions above are thus `foo[0]::bar[0]`
//!   and `foo[0]::bar[1]`. In order to incorporate this disambiguation
//!   information into the symbol name too, these indices are fed into the
//!   symbol hash, so that the above two symbols would end up with different
//!   hash values.
//!
//! The two measures described above suffice to avoid intra-crate conflicts. In
//! order to also avoid inter-crate conflicts two more measures are taken:
//!
//! - The name of the crate containing the symbol is prepended to the symbol
//!   name, i.e., symbols are "crate qualified". For example, a function `foo` in
//!   module `bar` in crate `baz` would get a symbol name like
//!   `baz::bar::foo::{hash}` instead of just `bar::foo::{hash}`. This avoids
//!   simple conflicts between functions from different crates.
//!
//! - In order to be able to also use symbols from two versions of the same
//!   crate (which naturally also have the same name), a stronger measure is
//!   required: The compiler accepts an arbitrary "disambiguator" value via the
//!   `-C metadata` command-line argument. This disambiguator is then fed into
//!   the symbol hash of every exported item. Consequently, the symbols in two
//!   identical crates but with different disambiguators are not in conflict
//!   with each other. This facility is mainly intended to be used by build
//!   tools like Cargo.
//!
//! A note on symbol name stability
//! -------------------------------
//! Previous versions of the compiler resorted to feeding NodeIds into the
//! symbol hash in order to disambiguate between items with the same path. The
//! current version of the name generation algorithm takes great care not to do
//! that, since NodeIds are notoriously unstable: A small change to the
//! code base will offset all NodeIds after the change and thus, much as using
//! the SVH in the hash, invalidate an unbounded number of symbol names. This
//! makes re-using previously compiled code for incremental compilation
//! virtually impossible. Thus, symbol hash generation exclusively relies on
//! DefPaths which are much more robust in the face of changes to the code base.

use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::hir::Node;
use rustc::hir::CodegenFnAttrFlags;
use rustc::hir::map::{DefPathData, DisambiguatedDefPathData};
use rustc::ich::NodeIdHashingMode;
use rustc::ty::print::{PrettyPrinter, Printer, Print};
use rustc::ty::query::Providers;
use rustc::ty::subst::{Kind, SubstsRef, UnpackedKind};
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc::util::common::record_time;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_mir::monomorphize::item::{InstantiationMode, MonoItem, MonoItemExt};
use rustc_mir::monomorphize::Instance;

use syntax_pos::symbol::InternedString;

use log::debug;

use std::fmt::{self, Write};
use std::mem::{self, discriminant};

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        symbol_name,

        ..*providers
    };
}

fn get_symbol_hash<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,

    // the DefId of the item this name is for
    def_id: DefId,

    // instance this name will be for
    instance: Instance<'tcx>,

    // type of the item, without any generic
    // parameters substituted; this is
    // included in the hash as a kind of
    // safeguard.
    item_type: Ty<'tcx>,

    // values for generic type parameters,
    // if any.
    substs: SubstsRef<'tcx>,
) -> u64 {
    debug!(
        "get_symbol_hash(def_id={:?}, parameters={:?})",
        def_id, substs
    );

    let mut hasher = StableHasher::<u64>::new();
    let mut hcx = tcx.create_stable_hashing_context();

    record_time(&tcx.sess.perf_stats.symbol_hash_time, || {
        // the main symbol name is not necessarily unique; hash in the
        // compiler's internal def-path, guaranteeing each symbol has a
        // truly unique path
        tcx.def_path_hash(def_id).hash_stable(&mut hcx, &mut hasher);

        // Include the main item-type. Note that, in this case, the
        // assertions about `needs_subst` may not hold, but this item-type
        // ought to be the same for every reference anyway.
        assert!(!item_type.has_erasable_regions());
        hcx.while_hashing_spans(false, |hcx| {
            hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
                item_type.hash_stable(hcx, &mut hasher);
            });
        });

        // If this is a function, we hash the signature as well.
        // This is not *strictly* needed, but it may help in some
        // situations, see the `run-make/a-b-a-linker-guard` test.
        if let ty::FnDef(..) = item_type.sty {
            item_type.fn_sig(tcx).hash_stable(&mut hcx, &mut hasher);
        }

        // also include any type parameters (for generic items)
        assert!(!substs.has_erasable_regions());
        assert!(!substs.needs_subst());
        substs.hash_stable(&mut hcx, &mut hasher);

        let is_generic = substs.non_erasable_generics().next().is_some();
        let avoid_cross_crate_conflicts =
            // If this is an instance of a generic function, we also hash in
            // the ID of the instantiating crate. This avoids symbol conflicts
            // in case the same instances is emitted in two crates of the same
            // project.
            is_generic ||

            // If we're dealing with an instance of a function that's inlined from
            // another crate but we're marking it as globally shared to our
            // compliation (aka we're not making an internal copy in each of our
            // codegen units) then this symbol may become an exported (but hidden
            // visibility) symbol. This means that multiple crates may do the same
            // and we want to be sure to avoid any symbol conflicts here.
            match MonoItem::Fn(instance).instantiation_mode(tcx) {
                InstantiationMode::GloballyShared { may_conflict: true } => true,
                _ => false,
            };

        if avoid_cross_crate_conflicts {
            let instantiating_crate = if is_generic {
                if !def_id.is_local() && tcx.sess.opts.share_generics() {
                    // If we are re-using a monomorphization from another crate,
                    // we have to compute the symbol hash accordingly.
                    let upstream_monomorphizations = tcx.upstream_monomorphizations_for(def_id);

                    upstream_monomorphizations
                        .and_then(|monos| monos.get(&substs).cloned())
                        .unwrap_or(LOCAL_CRATE)
                } else {
                    LOCAL_CRATE
                }
            } else {
                LOCAL_CRATE
            };

            (&tcx.original_crate_name(instantiating_crate).as_str()[..])
                .hash_stable(&mut hcx, &mut hasher);
            (&tcx.crate_disambiguator(instantiating_crate)).hash_stable(&mut hcx, &mut hasher);
        }

        // We want to avoid accidental collision between different types of instances.
        // Especially, VtableShim may overlap with its original instance without this.
        discriminant(&instance.def).hash_stable(&mut hcx, &mut hasher);
    });

    // 64 bits should be enough to avoid collisions.
    hasher.finish()
}

fn symbol_name(tcx: TyCtxt<'_, 'tcx, 'tcx>, instance: Instance<'tcx>) -> ty::SymbolName {
    ty::SymbolName {
        name: compute_symbol_name(tcx, instance),
    }
}

fn compute_symbol_name(tcx: TyCtxt<'_, 'tcx, 'tcx>, instance: Instance<'tcx>) -> InternedString {
    let def_id = instance.def_id();
    let substs = instance.substs;

    debug!("symbol_name(def_id={:?}, substs={:?})", def_id, substs);

    let hir_id = tcx.hir().as_local_hir_id(def_id);

    if def_id.is_local() {
        if tcx.plugin_registrar_fn(LOCAL_CRATE) == Some(def_id) {
            let disambiguator = tcx.sess.local_crate_disambiguator();
            return
                InternedString::intern(&tcx.sess.generate_plugin_registrar_symbol(disambiguator));
        }
        if tcx.proc_macro_decls_static(LOCAL_CRATE) == Some(def_id) {
            let disambiguator = tcx.sess.local_crate_disambiguator();
            return
                InternedString::intern(&tcx.sess.generate_proc_macro_decls_symbol(disambiguator));
        }
    }

    // FIXME(eddyb) Precompute a custom symbol name based on attributes.
    let is_foreign = if let Some(id) = hir_id {
        match tcx.hir().get_by_hir_id(id) {
            Node::ForeignItem(_) => true,
            _ => false,
        }
    } else {
        tcx.is_foreign_item(def_id)
    };

    let attrs = tcx.codegen_fn_attrs(def_id);
    if is_foreign {
        if let Some(name) = attrs.link_name {
            return name.as_interned_str();
        }
        // Don't mangle foreign items.
        return tcx.item_name(def_id).as_interned_str();
    }

    if let Some(name) = &attrs.export_name {
        // Use provided name
        return name.as_interned_str();
    }

    if attrs.flags.contains(CodegenFnAttrFlags::NO_MANGLE) {
        // Don't mangle
        return tcx.item_name(def_id).as_interned_str();
    }

    // We want to compute the "type" of this item. Unfortunately, some
    // kinds of items (e.g., closures) don't have an entry in the
    // item-type array. So walk back up the find the closest parent
    // that DOES have an entry.
    let mut ty_def_id = def_id;
    let instance_ty;
    loop {
        let key = tcx.def_key(ty_def_id);
        match key.disambiguated_data.data {
            DefPathData::TypeNs(_) | DefPathData::ValueNs(_) => {
                instance_ty = tcx.type_of(ty_def_id);
                break;
            }
            _ => {
                // if we're making a symbol for something, there ought
                // to be a value or type-def or something in there
                // *somewhere*
                ty_def_id.index = key.parent.unwrap_or_else(|| {
                    bug!(
                        "finding type for {:?}, encountered def-id {:?} with no \
                         parent",
                        def_id,
                        ty_def_id
                    );
                });
            }
        }
    }

    // Erase regions because they may not be deterministic when hashed
    // and should not matter anyhow.
    let instance_ty = tcx.erase_regions(&instance_ty);

    let hash = get_symbol_hash(tcx, def_id, instance, instance_ty, substs);

    let mut printer = SymbolPrinter {
        tcx,
        path: SymbolPath::new(),
        keep_within_component: false,
    }.print_def_path(def_id, &[]).unwrap();

    if instance.is_vtable_shim() {
        let _ = printer.write_str("{{vtable-shim}}");
    }

    InternedString::intern(&printer.path.finish(hash))
}

// Follow C++ namespace-mangling style, see
// http://en.wikipedia.org/wiki/Name_mangling for more info.
//
// It turns out that on macOS you can actually have arbitrary symbols in
// function names (at least when given to LLVM), but this is not possible
// when using unix's linker. Perhaps one day when we just use a linker from LLVM
// we won't need to do this name mangling. The problem with name mangling is
// that it seriously limits the available characters. For example we can't
// have things like &T in symbol names when one would theoretically
// want them for things like impls of traits on that type.
//
// To be able to work on all platforms and get *some* reasonable output, we
// use C++ name-mangling.
#[derive(Debug)]
struct SymbolPath {
    result: String,
    temp_buf: String,
}

impl SymbolPath {
    fn new() -> Self {
        let mut result = SymbolPath {
            result: String::with_capacity(64),
            temp_buf: String::with_capacity(16),
        };
        result.result.push_str("_ZN"); // _Z == Begin name-sequence, N == nested
        result
    }

    fn finalize_pending_component(&mut self) {
        if !self.temp_buf.is_empty() {
            let _ = write!(self.result, "{}{}", self.temp_buf.len(), self.temp_buf);
            self.temp_buf.clear();
        }
    }

    fn finish(mut self, hash: u64) -> String {
        self.finalize_pending_component();
        // E = end name-sequence
        let _ = write!(self.result, "17h{:016x}E", hash);
        self.result
    }
}

struct SymbolPrinter<'a, 'tcx> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    path: SymbolPath,

    // When `true`, `finalize_pending_component` isn't used.
    // This is needed when recursing into `path_qualified`,
    // or `path_generic_args`, as any nested paths are
    // logically within one component.
    keep_within_component: bool,
}

// HACK(eddyb) this relies on using the `fmt` interface to get
// `PrettyPrinter` aka pretty printing of e.g. types in paths,
// symbol names should have their own printing machinery.

impl Printer<'tcx, 'tcx> for SymbolPrinter<'_, 'tcx> {
    type Error = fmt::Error;

    type Path = Self;
    type Region = Self;
    type Type = Self;
    type DynExistential = Self;

    fn tcx(&'a self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.tcx
    }

    fn print_region(
        self,
        _region: ty::Region<'_>,
    ) -> Result<Self::Region, Self::Error> {
        Ok(self)
    }

    fn print_type(
        self,
        ty: Ty<'tcx>,
    ) -> Result<Self::Type, Self::Error> {
        match ty.sty {
            // Print all nominal types as paths (unlike `pretty_print_type`).
            ty::FnDef(def_id, substs) |
            ty::Opaque(def_id, substs) |
            ty::Projection(ty::ProjectionTy { item_def_id: def_id, substs }) |
            ty::UnnormalizedProjection(ty::ProjectionTy { item_def_id: def_id, substs }) |
            ty::Closure(def_id, ty::ClosureSubsts { substs }) |
            ty::Generator(def_id, ty::GeneratorSubsts { substs }, _) => {
                self.print_def_path(def_id, substs)
            }
            _ => self.pretty_print_type(ty),
        }
    }

    fn print_dyn_existential(
        mut self,
        predicates: &'tcx ty::List<ty::ExistentialPredicate<'tcx>>,
    ) -> Result<Self::DynExistential, Self::Error> {
        let mut first = false;
        for p in predicates {
            if !first {
                write!(self, "+")?;
            }
            first = false;
            self = p.print(self)?;
        }
        Ok(self)
    }

    fn path_crate(
        mut self,
        cnum: CrateNum,
    ) -> Result<Self::Path, Self::Error> {
        self.write_str(&self.tcx.original_crate_name(cnum).as_str())?;
        Ok(self)
    }
    fn path_qualified(
        self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        // Similar to `pretty_path_qualified`, but for the other
        // types that are printed as paths (see `print_type` above).
        match self_ty.sty {
            ty::FnDef(..) |
            ty::Opaque(..) |
            ty::Projection(_) |
            ty::UnnormalizedProjection(_) |
            ty::Closure(..) |
            ty::Generator(..)
                if trait_ref.is_none() =>
            {
                self.print_type(self_ty)
            }

            _ => self.pretty_path_qualified(self_ty, trait_ref)
        }
    }

    fn path_append_impl(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        _disambiguated_data: &DisambiguatedDefPathData,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self.pretty_path_append_impl(
            |mut cx| {
                cx = print_prefix(cx)?;

                if cx.keep_within_component {
                    // HACK(eddyb) print the path similarly to how `FmtPrinter` prints it.
                    cx.write_str("::")?;
                } else {
                    cx.path.finalize_pending_component();
                }

                Ok(cx)
            },
            self_ty,
            trait_ref,
        )
    }
    fn path_append(
        mut self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<Self::Path, Self::Error> {
        self = print_prefix(self)?;

        // Skip `::{{constructor}}` on tuple/unit structs.
        match disambiguated_data.data {
            DefPathData::Ctor => return Ok(self),
            _ => {}
        }

        if self.keep_within_component {
            // HACK(eddyb) print the path similarly to how `FmtPrinter` prints it.
            self.write_str("::")?;
        } else {
            self.path.finalize_pending_component();
        }

        self.write_str(&disambiguated_data.data.as_interned_str().as_str())?;
        Ok(self)
    }
    fn path_generic_args(
        mut self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        args: &[Kind<'tcx>],
    )  -> Result<Self::Path, Self::Error> {
        self = print_prefix(self)?;

        let args = args.iter().cloned().filter(|arg| {
            match arg.unpack() {
                UnpackedKind::Lifetime(_) => false,
                _ => true,
            }
        });

        if args.clone().next().is_some() {
            self.generic_delimiters(|cx| cx.comma_sep(args))
        } else {
            Ok(self)
        }
    }
}

impl PrettyPrinter<'tcx, 'tcx> for SymbolPrinter<'_, 'tcx> {
    fn region_should_not_be_omitted(
        &self,
        _region: ty::Region<'_>,
    ) -> bool {
        false
    }
    fn comma_sep<T>(
        mut self,
        mut elems: impl Iterator<Item = T>,
    ) -> Result<Self, Self::Error>
        where T: Print<'tcx, 'tcx, Self, Output = Self, Error = Self::Error>
    {
        if let Some(first) = elems.next() {
            self = first.print(self)?;
            for elem in elems {
                self.write_str(",")?;
                self = elem.print(self)?;
            }
        }
        Ok(self)
    }

    fn generic_delimiters(
        mut self,
        f: impl FnOnce(Self) -> Result<Self, Self::Error>,
    ) -> Result<Self, Self::Error> {
        write!(self, "<")?;

        let kept_within_component =
            mem::replace(&mut self.keep_within_component, true);
        self = f(self)?;
        self.keep_within_component = kept_within_component;

        write!(self, ">")?;

        Ok(self)
    }
}

impl fmt::Write for SymbolPrinter<'_, '_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        // Name sanitation. LLVM will happily accept identifiers with weird names, but
        // gas doesn't!
        // gas accepts the following characters in symbols: a-z, A-Z, 0-9, ., _, $
        // NVPTX assembly has more strict naming rules than gas, so additionally, dots
        // are replaced with '$' there.

        for c in s.chars() {
            if self.path.temp_buf.is_empty() {
                match c {
                    'a'..='z' | 'A'..='Z' | '_' => {}
                    _ => {
                        // Underscore-qualify anything that didn't start as an ident.
                        self.path.temp_buf.push('_');
                    }
                }
            }
            match c {
                // Escape these with $ sequences
                '@' => self.path.temp_buf.push_str("$SP$"),
                '*' => self.path.temp_buf.push_str("$BP$"),
                '&' => self.path.temp_buf.push_str("$RF$"),
                '<' => self.path.temp_buf.push_str("$LT$"),
                '>' => self.path.temp_buf.push_str("$GT$"),
                '(' => self.path.temp_buf.push_str("$LP$"),
                ')' => self.path.temp_buf.push_str("$RP$"),
                ',' => self.path.temp_buf.push_str("$C$"),

                '-' | ':' | '.' if self.tcx.has_strict_asm_symbol_naming() => {
                    // NVPTX doesn't support these characters in symbol names.
                    self.path.temp_buf.push('$')
                }

                // '.' doesn't occur in types and functions, so reuse it
                // for ':' and '-'
                '-' | ':' => self.path.temp_buf.push('.'),

                // These are legal symbols
                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '.' | '$' => self.path.temp_buf.push(c),

                _ => {
                    self.path.temp_buf.push('$');
                    for c in c.escape_unicode().skip(1) {
                        match c {
                            '{' => {}
                            '}' => self.path.temp_buf.push('$'),
                            c => self.path.temp_buf.push(c),
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
