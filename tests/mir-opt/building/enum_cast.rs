//@ compile-flags: -Zmir-opt-level=0
// skip-filecheck
// EMIT_MIR enum_cast.foo.built.after.mir
// EMIT_MIR enum_cast.bar.built.after.mir
// EMIT_MIR enum_cast.boo.built.after.mir
// EMIT_MIR enum_cast.far.built.after.mir

// Previously MIR building included range `Assume`s in the MIR statements,
// which these tests demonstrated, but now that we have range metadata on
// parameters in LLVM (in addition to !range metadata on loads) the impact
// of the extra volume of MIR is worse than its value.
// Thus these are now about the discriminant type and the cast type,
// both of which might be different from the backend type of the tag.

enum Foo {
    A,
}

enum Bar {
    A,
    B,
}

#[repr(u8)]
enum Boo {
    A,
    B,
}

#[repr(i16)]
enum Far {
    A,
    B,
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

enum NotStartingAtZero {
    A = 4,
    B = 6,
    C = 8,
}

// EMIT_MIR enum_cast.offsetty.built.after.mir
fn offsetty(x: NotStartingAtZero) -> u32 {
    x as u32
}

fn main() {}
