//! This module contains `HashStable` implementations for various data types
//! from rustc::middle::cstore in no particular order.

use middle;

impl_stable_hash_for!(enum middle::cstore::DepKind {
    UnexportedMacrosOnly,
    MacrosOnly,
    Implicit,
    Explicit
});

impl_stable_hash_for!(enum middle::cstore::NativeLibraryKind {
    NativeStatic,
    NativeStaticNobundle,
    NativeFramework,
    NativeUnknown
});

impl_stable_hash_for!(struct middle::cstore::NativeLibrary {
    kind,
    name,
    cfg,
    foreign_module,
    wasm_import_module
});

impl_stable_hash_for!(struct middle::cstore::ForeignModule {
    foreign_items,
    def_id
});

impl_stable_hash_for!(enum middle::cstore::LinkagePreference {
    RequireDynamic,
    RequireStatic
});

impl_stable_hash_for!(struct middle::cstore::ExternCrate {
    src,
    span,
    path_len,
    direct
});

impl_stable_hash_for!(enum middle::cstore::ExternCrateSource {
    Extern(def_id),
    Use,
    Path,
});

impl_stable_hash_for!(struct middle::cstore::CrateSource {
    dylib,
    rlib,
    rmeta
});
