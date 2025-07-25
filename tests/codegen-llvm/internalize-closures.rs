//@ compile-flags: -C no-prepopulate-passes -Zmir-opt-level=0

pub fn main() {
    // We want to make sure that closures get 'internal' linkage instead of
    // 'weak_odr' when they are not shared between codegen units
    // FIXME(eddyb) `legacy` mangling uses `{{closure}}`, while `v0`
    // uses `{closure#0}`, switch to the latter once `legacy` is gone.
    // CHECK-LABEL: ; internalize_closures::main::{{.*}}closure
    // CHECK-NEXT: ; Function Attrs:
    // CHECK-NEXT: define internal
    let c = |x: i32| x + 1;
    let _ = c(1);
}
