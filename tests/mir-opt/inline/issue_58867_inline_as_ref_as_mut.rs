//@ compile-flags: -C debuginfo=full

// EMIT_MIR issue_58867_inline_as_ref_as_mut.a.Inline.after.mir
pub fn a<T>(x: &mut [T]) -> &mut [T] {
    // CHECK-LABEL: fn a(
    // CHECK: (inlined <[T] as AsMut<[T]>>::as_mut)
    x.as_mut()
}

// EMIT_MIR issue_58867_inline_as_ref_as_mut.b.Inline.after.mir
pub fn b<T>(x: &mut Box<T>) -> &mut T {
    // CHECK-LABEL: fn b(
    // CHECK: (inlined <Box<T> as AsMut<T>>::as_mut)
    x.as_mut()
}

// EMIT_MIR issue_58867_inline_as_ref_as_mut.c.Inline.after.mir
pub fn c<T>(x: &[T]) -> &[T] {
    // CHECK-LABEL: fn c(
    // CHECK: (inlined <[T] as AsRef<[T]>>::as_ref)
    x.as_ref()
}

// EMIT_MIR issue_58867_inline_as_ref_as_mut.d.Inline.after.mir
pub fn d<T>(x: &Box<T>) -> &T {
    // CHECK-LABEL: fn d(
    // CHECK: (inlined <Box<T> as AsRef<T>>::as_ref)
    x.as_ref()
}

fn main() {
    let mut boxed = Box::new(1);
    println!("{:?}", a(&mut [1]));
    println!("{:?}", b(&mut boxed));
    println!("{:?}", c(&[1]));
    println!("{:?}", d(&boxed));
}
