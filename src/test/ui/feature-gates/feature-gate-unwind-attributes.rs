// compile-flags: -C no-prepopulate-passes -Cpasses=name-anon-globals

#![crate_type = "lib"]

extern {
// CHECK: Function Attrs: nounwind
// CHECK-NEXT: declare void @extern_fn
    fn extern_fn();
// CHECK-NOT: Function Attrs: nounwind
// CHECK: declare void @unwinding_extern_fn
    #[unwind(allowed)] //~ ERROR #[unwind] is experimental
    fn unwinding_extern_fn();
}

pub unsafe fn force_declare() {
    extern_fn();
    unwinding_extern_fn();
}
