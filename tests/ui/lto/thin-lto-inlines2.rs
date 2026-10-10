//@ run-pass

//@ compile-flags: -C codegen-units=8 -O -C lto=thin
//@ aux-build:thin-lto-inlines-aux.rs
//@ no-prefer-dynamic
//@ ignore-emscripten can't inspect instructions on emscripten
//@ ignore-backends: gcc

// We want to assert here that ThinLTO will inline across codegen units. There's
// not really a great way to do that in general so we sort of hack around it by
// praying two functions go into separate codegen units and then assuming that
// if inlining *doesn't* happen the first byte of the functions will differ.

#![allow(function_casts_as_integer)]

extern crate thin_lto_inlines_aux as bar;

pub fn foo() -> u32 {
    bar::bar()
}

fn main() {
    println!("{} {}", foo(), bar::bar());

    unsafe {
        let mut foo = foo as usize as *const u8;
        let bar = bar::bar as usize as *const u8;

        // cf-protection puts a NOP-like instruction, ENDBR64, at the start
        // of each function, but that's not present for the inlined version.
        // Skip that if present.
        if std::slice::from_raw_parts(foo, 4) == [0xf3, 0x0f, 0x1e, 0xfa] {
            foo = foo.add(4);
        }

        assert_eq!(*foo, *bar);
    }
}
