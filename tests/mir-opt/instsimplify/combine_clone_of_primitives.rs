//@ test-mir-pass: InstSimplify-after-simplifycfg
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// EMIT_MIR combine_clone_of_primitives.{impl#0}-clone.InstSimplify-after-simplifycfg.diff
#[derive(Clone)]
struct MyThing<T> {
    v: T,
    i: u64,
    a: [f32; 3],
}

// CHECK-LABEL: ::clone(
// CHECK: <T as Clone>::clone(
// CHECK-NOT: <u64 as Clone>::clone(
// CHECK-NOT: <[f32; 3] as Clone>::clone(

fn main() {
    let x = MyThing::<i16> { v: 2, i: 3, a: [0.0; 3] };
    let y = x.clone();

    assert_eq!(y.v, 2);
    assert_eq!(y.i, 3);
    assert_eq!(y.a, [0.0; 3]);
}
