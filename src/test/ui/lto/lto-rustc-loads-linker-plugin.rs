// compile-flags: -C lto
// aux-build:lto-rustc-loads-linker-plugin.rs
// run-pass
// no-prefer-dynamic

// This test ensures that if a dependency was compiled with
// `-Clinker-plugin-lto` then we can compile with `-Clto` and still link against
// that upstream rlib. This should work because LTO implies we're not actually
// linking against upstream rlibs since we're generating the object code
// locally. This test will fail if rustc can't find bytecode in rlibs compiled
// with `-Clinker-plugin-lto`.

extern crate lto_rustc_loads_linker_plugin;

fn main() {
    lto_rustc_loads_linker_plugin::foo();
}
