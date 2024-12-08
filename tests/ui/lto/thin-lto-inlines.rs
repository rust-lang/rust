//@ run-pass

//@ compile-flags: -Z thinlto -C codegen-units=8 -O
//@ ignore-emscripten can't inspect instructions on emscripten

// We want to assert here that ThinLTO will inline across codegen units. There's
// not really a great way to do that in general so we sort of hack around it by
// praying two functions go into separate codegen units and then assuming that
// if inlining *doesn't* happen the first byte of the functions will differ.

pub fn foo() -> u32 {
    bar::bar()
}

mod bar {
    pub fn bar() -> u32 {
        3
    }
}

fn main() {
    println!("{} {}", foo(), bar::bar());

    unsafe {
        let foo = foo as usize as *const u8;
        let bar = bar::bar as usize as *const u8;

        assert_eq!(*foo, *bar);
    }
}
