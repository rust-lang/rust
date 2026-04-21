// Validate the cross-global-crate trait-cast check from the RFC.
//
// Build layout:
//   common.rs      -> rlib (not a global crate)
//   cdylib_a.rs    -> cdylib (global crate)
//   cdylib_b.rs    -> cdylib (global crate)
//   program.rs     -> bin    (global crate), links both cdylibs via
//                             #[link(name = "cdylib_{a,b}")].
//
// Because each of the three global crates independently computes its
// trait-cast layout over disjoint sets of concrete types, every graph
// assigns `dyn Sub` the same slot index (0). Only the per-crate
// `global_crate_id` token distinguishes them. `program` exercises the
// full 3x3 matrix of (object-origin-crate, cast-site-crate) pairs and
// asserts that off-diagonal pairs are rejected with
// `TraitCastError::ForeignTraitGraph` while diagonal pairs succeed.

//@ ignore-cross-compile
// The test produces a binary that loads two cdylibs at runtime.

//@ needs-target-std
// program.rs links against std implicitly via core/alloc; running it on
// a target without std is not meaningful.

use run_make_support::{run, rustc};

fn main() {
    // Shared rlib — provides `Root`/`Sub` and the `RootRef` FFI carrier.
    rustc().input("common.rs").run();

    // Two independent global cdylibs. Each carries its own copy of the
    // trait-cast tables and its own address-significant
    // `global_crate_id` allocation.
    rustc().input("cdylib_a.rs").run();
    rustc().input("cdylib_b.rs").run();

    // Bin — also a global crate. `-L .` lets rustc find `libcommon.rlib`,
    // `libcdylib_a.so`, and `libcdylib_b.so` in the cwd.
    rustc().input("program.rs").run();

    // `run()` sets LD_LIBRARY_PATH (or platform equivalent) to include
    // the cwd, so the two cdylibs are resolved at load time.
    run("program");
}
