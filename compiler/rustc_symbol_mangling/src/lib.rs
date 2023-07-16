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

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(never_type)]
#![recursion_limit = "256"]
#![allow(rustc::potential_query_instability)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate rustc_middle;

#[macro_use]
extern crate tracing;

use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_fluent_macro::fluent_messages;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::mir::mono::{InstantiationMode, MonoItem};
use rustc_middle::query::Providers;
use rustc_middle::ty::GenericArgsRef;
use rustc_middle::ty::{self, Instance, TyCtxt};
use rustc_session::config::SymbolManglingVersion;

mod legacy;
mod v0;

pub mod errors;
pub mod test;
pub mod typeid;

fluent_messages! { "../messages.ftl" }

/// This function computes the symbol name for the given `instance` and the
/// given instantiating crate. That is, if you know that instance X is
/// instantiated in crate Y, this is the symbol name this instance would have.
pub fn symbol_name_for_instance_in_crate<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    instantiating_crate: CrateNum,
) -> String {
    compute_symbol_name(tcx, instance, || instantiating_crate)
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { symbol_name: symbol_name_provider, ..*providers };
}

// The `symbol_name` query provides the symbol name for calling a given
// instance from the local crate. In particular, it will also look up the
// correct symbol name of instances from upstream crates.
fn symbol_name_provider<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> ty::SymbolName<'tcx> {
    let symbol_name = compute_symbol_name(tcx, instance, || {
        // This closure determines the instantiating crate for instances that
        // need an instantiating-crate-suffix for their symbol name, in order
        // to differentiate between local copies.
        if is_generic(instance.args) {
            // For generics we might find re-usable upstream instances. If there
            // is one, we rely on the symbol being instantiated locally.
            instance.upstream_monomorphization(tcx).unwrap_or(LOCAL_CRATE)
        } else {
            // For non-generic things that need to avoid naming conflicts, we
            // always instantiate a copy in the local crate.
            LOCAL_CRATE
        }
    });

    ty::SymbolName::new(tcx, &symbol_name)
}

pub fn typeid_for_trait_ref<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::PolyExistentialTraitRef<'tcx>,
) -> String {
    v0::mangle_typeid_for_trait_ref(tcx, trait_ref)
}

/// Computes the symbol name for the given instance. This function will call
/// `compute_instantiating_crate` if it needs to factor the instantiating crate
/// into the symbol name.
fn compute_symbol_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    compute_instantiating_crate: impl FnOnce() -> CrateNum,
) -> String {
    let def_id = instance.def_id();
    let args = instance.args;

    debug!("symbol_name(def_id={:?}, args={:?})", def_id, args);

    if let Some(def_id) = def_id.as_local() {
        if tcx.proc_macro_decls_static(()) == Some(def_id) {
            let stable_crate_id = tcx.sess.local_stable_crate_id();
            return tcx.sess.generate_proc_macro_decls_symbol(stable_crate_id);
        }
    }

    // FIXME(eddyb) Precompute a custom symbol name based on attributes.
    let attrs = if tcx.def_kind(def_id).has_codegen_attrs() {
        tcx.codegen_fn_attrs(def_id)
    } else {
        CodegenFnAttrs::EMPTY
    };

    // Foreign items by default use no mangling for their symbol name. There's a
    // few exceptions to this rule though:
    //
    // * This can be overridden with the `#[link_name]` attribute
    //
    // * On the wasm32 targets there is a bug (or feature) in LLD [1] where the
    //   same-named symbol when imported from different wasm modules will get
    //   hooked up incorrectly. As a result foreign symbols, on the wasm target,
    //   with a wasm import module, get mangled. Additionally our codegen will
    //   deduplicate symbols based purely on the symbol name, but for wasm this
    //   isn't quite right because the same-named symbol on wasm can come from
    //   different modules. For these reasons if `#[link(wasm_import_module)]`
    //   is present we mangle everything on wasm because the demangled form will
    //   show up in the `wasm-import-name` custom attribute in LLVM IR.
    //
    // [1]: https://bugs.llvm.org/show_bug.cgi?id=44316
    if tcx.is_foreign_item(def_id)
        && (!tcx.sess.target.is_like_wasm
            || !tcx.wasm_import_module_map(def_id.krate).contains_key(&def_id))
    {
        if let Some(name) = attrs.link_name {
            return name.to_string();
        }
        return tcx.item_name(def_id).to_string();
    }

    if let Some(name) = attrs.export_name {
        // Use provided name
        return name.to_string();
    }

    if attrs.flags.contains(CodegenFnAttrFlags::NO_MANGLE) {
        // Don't mangle
        return tcx.item_name(def_id).to_string();
    }

    // If we're dealing with an instance of a function that's inlined from
    // another crate but we're marking it as globally shared to our
    // compilation (aka we're not making an internal copy in each of our
    // codegen units) then this symbol may become an exported (but hidden
    // visibility) symbol. This means that multiple crates may do the same
    // and we want to be sure to avoid any symbol conflicts here.
    let is_globally_shared_function = matches!(
        tcx.def_kind(instance.def_id()),
        DefKind::Fn | DefKind::AssocFn | DefKind::Closure | DefKind::Generator | DefKind::Ctor(..)
    ) && matches!(
        MonoItem::Fn(instance).instantiation_mode(tcx),
        InstantiationMode::GloballyShared { may_conflict: true }
    );

    // If this is an instance of a generic function, we also hash in
    // the ID of the instantiating crate. This avoids symbol conflicts
    // in case the same instances is emitted in two crates of the same
    // project.
    let avoid_cross_crate_conflicts = is_generic(args) || is_globally_shared_function;

    let instantiating_crate = avoid_cross_crate_conflicts.then(compute_instantiating_crate);

    // Pick the crate responsible for the symbol mangling version, which has to:
    // 1. be stable for each instance, whether it's being defined or imported
    // 2. obey each crate's own `-C symbol-mangling-version`, as much as possible
    // We solve these as follows:
    // 1. because symbol names depend on both `def_id` and `instantiating_crate`,
    // both their `CrateNum`s are stable for any given instance, so we can pick
    // either and have a stable choice of symbol mangling version
    // 2. we favor `instantiating_crate` where possible (i.e. when `Some`)
    let mangling_version_crate = instantiating_crate.unwrap_or(def_id.krate);
    let mangling_version = if mangling_version_crate == LOCAL_CRATE {
        tcx.sess.opts.get_symbol_mangling_version()
    } else {
        tcx.symbol_mangling_version(mangling_version_crate)
    };

    let symbol = match mangling_version {
        SymbolManglingVersion::Legacy => legacy::mangle(tcx, instance, instantiating_crate),
        SymbolManglingVersion::V0 => v0::mangle(tcx, instance, instantiating_crate),
    };

    debug_assert!(
        rustc_demangle::try_demangle(&symbol).is_ok(),
        "compute_symbol_name: `{symbol}` cannot be demangled"
    );

    symbol
}

fn is_generic(args: GenericArgsRef<'_>) -> bool {
    args.non_erasable_generics().next().is_some()
}
