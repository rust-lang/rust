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

extern crate thin_lto_inlines_aux as bar;

pub fn foo() -> u32 {
    bar::bar()
}

fn main() {
    println!("{} {}", foo(), bar::bar());

    unsafe {
        let foo = foo as usize as *const u8;
        let bar = bar::bar as usize as *const u8;

        assert_eq!(*foo, *bar);
    }
}
