// compile-flags: -C no-prepopulate-passes

pub fn main() {

    // We want to make sure that closures get 'internal' linkage instead of
    // 'weak_odr' when they are not shared between codegen units
    // CHECK: define internal {{.*}}_ZN20internalize_closures4main{{.*}}$u7b$$u7b$closure$u7d$$u7d$
    let c = |x:i32| { x + 1 };
    let _ = c(1);
}
