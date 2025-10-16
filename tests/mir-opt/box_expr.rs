//@ test-mir-pass: ElaborateDrops
//@ needs-unwind

#![feature(rustc_attrs, liballoc_internals)]

// EMIT_MIR box_expr.move_from_inner.ElaborateDrops.diff
fn move_from_inner() {
    // CHECK-LABEL: fn move_from_inner(
    // CHECK:   debug x => [[x:_.*]];
    // CHECK:   [[box:_.*]] = ShallowInitBox(
    // CHECK:   [[ptr:_.*]] = copy (([[box]].0: std::ptr::Unique<S>).0: std::ptr::NonNull<S>) as *const S (Transmute);
    // CHECK:   (*[[ptr]]) = S::new() -> [return: [[ret:bb.*]], unwind: [[unwind:bb.*]]];
    // CHECK: [[ret]]: {
    // CHECK:   [[x]] = move [[box]];
    // CHECK:   [[ptr:_.*]] = copy (([[x]].0: std::ptr::Unique<S>).0: std::ptr::NonNull<S>) as *const S (Transmute);
    // CHECK:   [[inner:_.*]] = move (*[[ptr]]);
    // CHECK:   std::mem::drop::<S>(move [[inner]]) -> [return: [[ret2:bb.*]], unwind: [[unwind2:bb.*]]];
    // CHECK: [[ret2]]: {
    // CHECK:   [[boxptr:_.*]] = &mut [[x]];
    // CHECK:   <Box<S> as Drop>::drop(move [[boxptr]]) -> [return: [[ret3:bb.*]], unwind: [[unwind3:bb.*]]];
    // CHECK: [[unwind2]] (cleanup): {
    // CHECK:   [[boxptr3:_.*]] = &mut [[x]];
    // CHECK:   <Box<S> as Drop>::drop(move [[boxptr3]])
    // CHECK: [[unwind3]] (cleanup): {
    // CHECK:   resume;
    // CHECK: [[unwind]] (cleanup): {
    // CHECK:   [[boxptr2:_.*]] = &mut [[box]];
    // CHECK:   <Box<S> as Drop>::drop(move [[boxptr2]])

    let x = std::boxed::box_new(S::new());
    drop(*x);
}

// EMIT_MIR box_expr.main.ElaborateDrops.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK:   [[ptr:_.*]] = move {{_.*}} as *const S (Transmute);
    // CHECK:   [[nonnull:_.*]] = NonNull::<S> { pointer: move [[ptr]] };
    // CHECK:   [[unique:_.*]] = Unique::<S> { pointer: move [[nonnull]], _marker: const PhantomData::<S> };
    // CHECK:   [[box:_.*]] = Box::<S>(move [[unique]], const std::alloc::Global);
    // CHECK:   [[ptr:_.*]] = copy (([[box]].0: std::ptr::Unique<S>).0: std::ptr::NonNull<S>) as *const S (Transmute);
    // CHECK:   (*[[ptr]]) = S::new() -> [return: [[ret:bb.*]], unwind: [[unwind:bb.*]]];
    // CHECK: [[ret]]: {
    // CHECK:   [[box2:_.*]] = move [[box]];
    // CHECK:   [[box3:_.*]] = move [[box2]];
    // CHECK:   std::mem::drop::<Box<S>>(move [[box3]])
    // CHECK: [[unwind]] (cleanup): {
    // CHECK:   [[boxref:_.*]] = &mut [[box]];
    // CHECK:   <Box<S> as Drop>::drop(move [[boxref]])

    let x = std::boxed::box_new(S::new());
    drop(x);
}

struct S;

impl S {
    fn new() -> Self {
        S
    }
}

impl Drop for S {
    fn drop(&mut self) {
        println!("splat!");
    }
}
