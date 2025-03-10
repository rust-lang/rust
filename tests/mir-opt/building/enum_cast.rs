// EMIT_MIR enum_cast.foo.built.after.mir
// EMIT_MIR enum_cast.bar.built.after.mir
// EMIT_MIR enum_cast.boo.built.after.mir
// EMIT_MIR enum_cast.far.built.after.mir

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

// CHECK-LABEL: fn foo(
fn foo(foo: Foo) -> usize {
    // CHECK: _0 = const 0_usize;
    foo as usize
}

// CHECK-LABEL: fn bar(
fn bar(bar: Bar) -> usize {
    // CHECK: _2 = copy _1 as u8 (Transmute);
    // CHECK: _0 = move _2 as usize (IntToInt);
    bar as usize
}

// CHECK-LABEL: fn boo(
fn boo(boo: Boo) -> usize {
    // CHECK: _2 = copy _1 as u8 (Transmute);
    // CHECK: _0 = move _2 as usize (IntToInt);
    boo as usize
}

// CHECK-LABEL: fn far(
fn far(far: Far) -> isize {
    // CHECK: _2 = copy _1 as i16 (Transmute);
    // CHECK: _0 = move _2 as isize (IntToInt);
    far as isize
}

#[derive(Copy, Clone)]
enum SingleVariantWithCustomDiscriminant {
    FourtyTwo = 42,
}

#[derive(Copy, Clone)]
#[repr(u16)]
enum SingleVariantWithCustomDiscriminantAndRepr {
    FourtyTwo = 42,
}

// EMIT_MIR enum_cast.custom_single_variant.built.after.mir
// CHECK-LABEL: fn custom_single_variant(
#[rustfmt::skip] // No, I don't want the comments on the same lines with the commas
fn custom_single_variant(
    a: SingleVariantWithCustomDiscriminant,
    b: SingleVariantWithCustomDiscriminantAndRepr,
) -> impl Sized {
    (
        a as isize,
        a as usize,
        a as i16,
        a as u16,
        // CHECK: [[REPR:_.+]] = copy _2 as u16 (Transmute)
        // CHECK: [[B0:_.+]] = copy [[REPR]] as isize (IntToInt)
        b as isize,
        // CHECK: [[B1:_.+]] = copy [[REPR]] as usize (IntToInt)
        b as usize,
        // CHECK: [[B2:_.+]] = copy [[REPR]] as i16 (IntToInt)
        b as i16,
        b as u16,
    )
    // CHECK: _0 = (const 42_isize, const 42_usize, const 42_i16, const 42_u16,
    // CHECK-SAME: move [[B0]], move [[B1]], move [[B2]], copy [[REPR]]);
}

// EMIT_MIR enum_cast.unreachable.built.after.mir
// CHECK-LABEL: fn unreachable
fn unreachable(x: std::convert::Infallible) -> u16 {
    // CHECK: debug x => const ZeroSized: Infallible;
    // CHECK: bb0: {
    // CHECK-NEXT: unreachable;
    // CHECK-NEXT: }
    x as _
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
// CHECK-LABEL: fn signy(
fn signy(x: SignedAroundZero) -> i16 {
    // CHECK: _0 = copy _1 as i16 (Transmute);
    x as i16
}

// EMIT_MIR enum_cast.unsigny.built.after.mir
// CHECK-LABEL: fn unsigny(
fn unsigny(x: UnsignedAroundZero) -> u16 {
    // CHECK: _0 = copy _1 as u16 (Transmute);
    x as u16
}

enum NotStartingAtZero {
    A = 4,
    B = 6,
    C = 8,
}

// EMIT_MIR enum_cast.offsetty.built.after.mir
// CHECK-LABEL: fn offsetty(
fn offsetty(x: NotStartingAtZero) -> u32 {
    // CHECK: _2 = copy _1 as u8 (Transmute);
    // CHECK: _0 = move _2 as u32 (IntToInt);
    x as u32
}

enum ReallyBigDiscr {
    One = 0x4321,
}

// CHECK-LABEL: fn really_big_discr(
fn really_big_discr(x: ReallyBigDiscr) -> u8 {
    // Better not ICE on this!
    // CHECK: _0 = const 33_u8;
    x as u8
}

// `align` gives this `Memory` ABI instead of `Scalar`
#[repr(align(8))]
enum Aligned {
    Zero = 0,
    One = 1,
}

// CHECK-LABEL: fn aligned_as(
fn aligned_as(x: Aligned) -> u8 {
    // CHECK: _2 = discriminant(_1);
    // CHECK: _0 = move _2 as u8 (IntToInt);
    x as u8
}

fn main() {}
