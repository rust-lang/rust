// EMIT_MIR enum_cast.foo.mir_map.0.mir
// EMIT_MIR enum_cast.bar.mir_map.0.mir
// EMIT_MIR enum_cast.boo.mir_map.0.mir

enum Foo {
    A
}

enum Bar {
    A, B
}

#[repr(u8)]
enum Boo {
    A, B
}

fn foo(foo: Foo) -> usize {
    foo as usize
}

fn bar(bar: Bar) -> usize {
    bar as usize
}

fn boo(boo: Boo) -> usize {
    boo as usize
}

// EMIT_MIR enum_cast.droppy.mir_map.0.mir
enum Droppy {
    A, B, C
}

impl Drop for Droppy {
    fn drop(&mut self) {}
}

fn droppy() {
    {
        let x = Droppy::C;
        // remove this entire test once `cenum_impl_drop_cast` becomes a hard error
        #[allow(cenum_impl_drop_cast)]
        let y = x as usize;
    }
    let z = Droppy::B;
}

fn main() {
}
