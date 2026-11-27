// When crates had different optimization levels, a bug caused
// incorrect symbol name generations. -Z share-generics could
// also fail to re-export upstream generics on multiple compile
// runs of the same dynamic library.

// This test repeatedly compiles an rlib and a dylib with these flags
// to check if this bug ever returns.

// See https://github.com/rust-lang/rust/pull/68277
// See https://github.com/rust-lang/rust/issues/64319
//@ ignore-cross-compile

use run_make_support::rustc;

fn main() {
    rustc().crate_type("rlib").input("foo.rs").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("3").run();
    rustc().crate_type("rlib").input("foo.rs").arg("-Zshare-generics=no").run();
    rustc().crate_type("dylib").input("bar.rs").arg("-Zshare-generics=no").run();
    rustc().crate_type("rlib").input("foo.rs").arg("-Zshare-generics=no").run();
    rustc().crate_type("dylib").input("bar.rs").arg("-Zshare-generics=yes").run();
    rustc().crate_type("rlib").input("foo.rs").arg("-Zshare-generics=yes").run();
    rustc().crate_type("dylib").input("bar.rs").arg("-Zshare-generics=no").run();
    rustc().crate_type("rlib").input("foo.rs").arg("-Zshare-generics=yes").run();
    rustc().crate_type("dylib").input("bar.rs").arg("-Zshare-generics=yes").run();
    rustc().crate_type("rlib").input("foo.rs").run();
    rustc().crate_type("dylib").input("bar.rs").run();
    rustc().crate_type("dylib").input("bar.rs").arg("-Zshare-generics=no").run();
    rustc().crate_type("dylib").input("bar.rs").arg("-Zshare-generics=yes").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("1").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("1").arg("-Zshare-generics=no").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("1").arg("-Zshare-generics=yes").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("2").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("2").arg("-Zshare-generics=no").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("2").arg("-Zshare-generics=yes").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("3").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("3").arg("-Zshare-generics=no").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("3").arg("-Zshare-generics=yes").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("s").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("s").arg("-Zshare-generics=no").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("s").arg("-Zshare-generics=yes").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("z").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("z").arg("-Zshare-generics=no").run();
    rustc().crate_type("dylib").input("bar.rs").opt_level("z").arg("-Zshare-generics=yes").run();
}
