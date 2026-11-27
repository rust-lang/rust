//@ needs-unwind
//@ compile-flags: -C no-prepopulate-passes -Copt-level=s -Znounwind-inference

// Only the monomorphization proves these bodies quiet. Each caller holds a live `Drop` local,
// which otherwise forces an `invoke`.
//
// CHECK: define {{.*}} @_RIN{{.*}}generic_quiet{{.*}} unnamed_addr [[QUIET_DEF:#[0-9]+]]
// CHECK: call noundef i32 @_RIN{{.*}}generic_quiet{{.*}} [[CALL_ATTRS:#[0-9]+]]
// CHECK: invoke noundef i32 @_RIN{{.*}}generic_panicky{{.*}}
// A tuple constructor is not a function, but it gets a body; bound to a local, the call names it.
// CHECK: define {{.*}} @_RNc{{.*}}Pair0{{.*}} unnamed_addr [[CTOR_DEF:#[0-9]+]]
// CHECK: call {{.*}} @_RNc{{.*}}Pair0{{.*}} [[CALL_ATTRS]]
// CHECK-DAG: attributes [[QUIET_DEF]] = { {{[^}]*}}nounwind{{[^}]*}} }
// CHECK-DAG: attributes [[CTOR_DEF]] = { {{[^}]*}}nounwind{{[^}]*}} }
// CHECK-DAG: attributes [[CALL_ATTRS]] = { {{[^}]*}}nounwind{{[^}]*}} }
#![crate_type = "lib"]

struct Guard(u32);

impl Drop for Guard {
    #[inline(never)]
    fn drop(&mut self) {
        std::hint::black_box(self.0);
    }
}

struct Boom;

impl Drop for Boom {
    #[inline(never)]
    fn drop(&mut self) {
        panic!("boom");
    }
}

#[inline(never)]
fn generic_quiet<T>(v: T) -> u32 {
    drop(v);
    std::hint::black_box(1)
}

#[inline(never)]
fn generic_panicky<T>(v: T) -> u32 {
    drop(v);
    std::hint::black_box(2)
}

#[no_mangle]
pub fn call_generic_quiet(x: u32) -> u32 {
    let _g = Guard(x);
    generic_quiet(Guard(x))
}

#[no_mangle]
pub fn call_generic_panicky(x: u32) -> u32 {
    let _g = Guard(x);
    generic_panicky(Boom)
}

pub struct Pair(u32, u32);

#[no_mangle]
pub fn call_ctor(x: u32) -> Pair {
    let _g = Guard(x);
    let c = Pair;
    c(x, 1)
}
