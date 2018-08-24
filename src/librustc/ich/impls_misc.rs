//! This module contains `HashStable` implementations for various data types
//! that don't fit into any of the other impls_xxx modules.

impl_stable_hash_for!(enum ::session::search_paths::PathKind {
    Native,
    Crate,
    Dependency,
    Framework,
    ExternFlag,
    All
});

impl_stable_hash_for!(enum ::rustc_target::spec::PanicStrategy {
    Abort,
    Unwind
});
