// skip-filecheck
// EMIT_MIR enum_cast.foo.built.after.mir
// EMIT_MIR enum_cast.bar.built.after.mir
// EMIT_MIR enum_cast.boo.built.after.mir
// EMIT_MIR enum_cast.far.built.after.mir

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

#[repr(i16)]
enum Far {
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

fn far(far: Far) -> isize {
    far as isize
}

// EMIT_MIR enum_cast.droppy.built.after.mir
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

#[repr(i16)]
enum SignedAroundZero {
    A = -2,
    B = 0,
    C = 2,
}

#[repr(u16)]
enum UnsignedAroundZero {
    A = 65535,
    B = 0,
    C = 1,
}

// EMIT_MIR enum_cast.signy.built.after.mir
fn signy(x: SignedAroundZero) -> i16 {
    x as i16
}

// EMIT_MIR enum_cast.unsigny.built.after.mir
fn unsigny(x: UnsignedAroundZero) -> u16 {
    // FIXME: This doesn't get an around-the-end range today, sadly.
    x as u16
}

enum NotStartingAtZero { A = 4, B = 6, C = 8 }

// EMIT_MIR enum_cast.offsetty.built.after.mir
fn offsetty(x: NotStartingAtZero) -> u32 {
    x as u32
}

fn main() {
}
