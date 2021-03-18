// Regression test for #29988

// compile-flags: -C no-prepopulate-passes
// only-x86_64
// ignore-windows

#[repr(C)]
struct S {
    f1: i32,
    f2: i32,
    f3: i32,
}

extern "C" {
    fn foo(s: S);
}

fn main() {
    let s = S { f1: 1, f2: 2, f3: 3 };
    unsafe {
        // CHECK: load { i64, i32 }, { i64, i32 }* {{.*}}, align 4
        // CHECK: call void @foo({ i64, i32 } {{.*}})
        foo(s);
    }
}
