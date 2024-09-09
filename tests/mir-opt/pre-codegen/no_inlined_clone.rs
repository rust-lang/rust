#![crate_type = "lib"]

// EMIT_MIR no_inlined_clone.{impl#0}-clone.PreCodegen.after.mir

// CHECK-LABEL: ::clone(
// CHECK-NOT: inlined clone::impls::<impl Clone for {{.*}}>::clone
// CHECK: return;

#[derive(Clone)]
struct Foo {
    a: i32,
}
