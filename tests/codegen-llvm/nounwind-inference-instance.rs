//@ needs-unwind
//@ compile-flags: -C no-prepopulate-passes
//@ revisions: SIZE SIZEMIN
//@[SIZE] compile-flags: -Copt-level=s
//@[SIZEMIN] compile-flags: -Copt-level=z

// These bodies have no proof at the declaration; only the monomorphization shows they are quiet.
// Each case sits in a function with a live `Drop` local, which otherwise forces an `invoke`.

// The parameter's type is only known here.
//
// CHECK: define {{.*}} @_RIN{{.*}}generic_quiet{{.*}} unnamed_addr [[QUIET_DEF:#[0-9]+]]
// CHECK: call noundef i32 @_RIN{{.*}}generic_quiet{{.*}} [[CALL_ATTRS:#[0-9]+]]
//
// The same shape reaching a panicking `Drop` keeps its `invoke`, so the checks above cannot pass
// vacuously.
// CHECK: invoke noundef i32 @_RIN{{.*}}generic_panicky{{.*}}
//
// A tuple constructor is not a function, but codegen emits a body for it. Bound to a local, the
// call names the constructor instead of becoming an assignment.
//
// CHECK: define {{.*}} @_RNc{{.*}}Pair0{{.*}} unnamed_addr [[CTOR_DEF:#[0-9]+]]
// CHECK: call {{.*}} @_RNc{{.*}}Pair0{{.*}} [[CALL_ATTRS]]
//
// Both call sites are quiet, so they name the same attribute group.
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
