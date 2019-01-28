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

use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::hir::Node;
use rustc::hir::CodegenFnAttrFlags;
use rustc::hir::map::definitions::DefPathData;
use rustc::ich::NodeIdHashingMode;
use rustc::ty::item_path::{self, ItemPathBuffer, RootMode};
use rustc::ty::query::Providers;
use rustc::ty::subst::Substs;
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc::util::common::record_time;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_mir::monomorphize::item::{InstantiationMode, MonoItem, MonoItemExt};
use rustc_mir::monomorphize::Instance;

use syntax_pos::symbol::Symbol;

use std::fmt::Write;
use std::mem::discriminant;

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        def_symbol_name,
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
    substs: &'tcx Substs<'tcx>,
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

        let is_generic = substs.types().next().is_some();
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

fn def_symbol_name<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> ty::SymbolName {
    let mut buffer = SymbolPathBuffer::new(tcx);
    item_path::with_forced_absolute_paths(|| {
        tcx.push_item_path(&mut buffer, def_id, false);
    });
    buffer.into_interned()
}

fn symbol_name<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, instance: Instance<'tcx>) -> ty::SymbolName {
    ty::SymbolName {
        name: Symbol::intern(&compute_symbol_name(tcx, instance)).as_interned_str(),
    }
}

fn compute_symbol_name<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, instance: Instance<'tcx>) -> String {
    let def_id = instance.def_id();
    let substs = instance.substs;

    debug!("symbol_name(def_id={:?}, substs={:?})", def_id, substs);

    let node_id = tcx.hir().as_local_node_id(def_id);

    if def_id.is_local() {
        if tcx.plugin_registrar_fn(LOCAL_CRATE) == Some(def_id) {
            let disambiguator = tcx.sess.local_crate_disambiguator();
            return tcx.sess.generate_plugin_registrar_symbol(disambiguator);
        }
        if tcx.proc_macro_decls_static(LOCAL_CRATE) == Some(def_id) {
            let disambiguator = tcx.sess.local_crate_disambiguator();
            return tcx.sess.generate_proc_macro_decls_symbol(disambiguator);
        }
    }

    // FIXME(eddyb) Precompute a custom symbol name based on attributes.
    let is_foreign = if let Some(id) = node_id {
        match tcx.hir().get(id) {
            Node::ForeignItem(_) => true,
            _ => false,
        }
    } else {
        tcx.is_foreign_item(def_id)
    };

    let attrs = tcx.codegen_fn_attrs(def_id);
    if is_foreign {
        if let Some(name) = attrs.link_name {
            return name.to_string();
        }
        // Don't mangle foreign items.
        return tcx.item_name(def_id).to_string();
    }

    if let Some(name) = &attrs.export_name {
        // Use provided name
        return name.to_string();
    }

    if attrs.flags.contains(CodegenFnAttrFlags::NO_MANGLE) {
        // Don't mangle
        return tcx.item_name(def_id).to_string();
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

    let mut buf = SymbolPathBuffer::from_interned(tcx.def_symbol_name(def_id), tcx);

    if instance.is_vtable_shim() {
        buf.push("{{vtable-shim}}");
    }

    buf.finish(hash)
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
struct SymbolPathBuffer {
    result: String,
    temp_buf: String,
    strict_naming: bool,
}

impl SymbolPathBuffer {
    fn new(tcx: TyCtxt<'_, '_, '_>) -> Self {
        let mut result = SymbolPathBuffer {
            result: String::with_capacity(64),
            temp_buf: String::with_capacity(16),
            strict_naming: tcx.has_strict_asm_symbol_naming(),
        };
        result.result.push_str("_ZN"); // _Z == Begin name-sequence, N == nested
        result
    }

    fn from_interned(symbol: ty::SymbolName, tcx: TyCtxt<'_, '_, '_>) -> Self {
        let mut result = SymbolPathBuffer {
            result: String::with_capacity(64),
            temp_buf: String::with_capacity(16),
            strict_naming: tcx.has_strict_asm_symbol_naming(),
        };
        result.result.push_str(&symbol.as_str());
        result
    }

    fn into_interned(self) -> ty::SymbolName {
        ty::SymbolName {
            name: Symbol::intern(&self.result).as_interned_str(),
        }
    }

    fn finish(mut self, hash: u64) -> String {
        // E = end name-sequence
        let _ = write!(self.result, "17h{:016x}E", hash);
        self.result
    }

    // Name sanitation. LLVM will happily accept identifiers with weird names, but
    // gas doesn't!
    // gas accepts the following characters in symbols: a-z, A-Z, 0-9, ., _, $
    // NVPTX assembly has more strict naming rules than gas, so additionally, dots
    // are replaced with '$' there.
    fn sanitize_and_append(&mut self, s: &str) {
        self.temp_buf.clear();

        for c in s.chars() {
            match c {
                // Escape these with $ sequences
                '@' => self.temp_buf.push_str("$SP$"),
                '*' => self.temp_buf.push_str("$BP$"),
                '&' => self.temp_buf.push_str("$RF$"),
                '<' => self.temp_buf.push_str("$LT$"),
                '>' => self.temp_buf.push_str("$GT$"),
                '(' => self.temp_buf.push_str("$LP$"),
                ')' => self.temp_buf.push_str("$RP$"),
                ',' => self.temp_buf.push_str("$C$"),

                '-' | ':' => if self.strict_naming {
                    // NVPTX doesn't support these characters in symbol names.
                    self.temp_buf.push('$')
                }
                else {
                    // '.' doesn't occur in types and functions, so reuse it
                    // for ':' and '-'
                    self.temp_buf.push('.')
                },

                '.' => if self.strict_naming {
                    self.temp_buf.push('$')
                }
                else {
                    self.temp_buf.push('.')
                },

                // These are legal symbols
                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '$' => self.temp_buf.push(c),

                _ => {
                    self.temp_buf.push('$');
                    for c in c.escape_unicode().skip(1) {
                        match c {
                            '{' => {}
                            '}' => self.temp_buf.push('$'),
                            c => self.temp_buf.push(c),
                        }
                    }
                }
            }
        }

        let need_underscore = {
            // Underscore-qualify anything that didn't start as an ident.
            !self.temp_buf.is_empty()
                && self.temp_buf.as_bytes()[0] != '_' as u8
                && !(self.temp_buf.as_bytes()[0] as char).is_xid_start()
        };

        let _ = write!(
            self.result,
            "{}",
            self.temp_buf.len() + (need_underscore as usize)
        );

        if need_underscore {
            self.result.push('_');
        }

        self.result.push_str(&self.temp_buf);
    }
}

impl ItemPathBuffer for SymbolPathBuffer {
    fn root_mode(&self) -> &RootMode {
        const ABSOLUTE: &RootMode = &RootMode::Absolute;
        ABSOLUTE
    }

    fn push(&mut self, text: &str) {
        self.sanitize_and_append(text);
    }
}
