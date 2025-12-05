//@ compile-flags: -C no-prepopulate-passes
// 32bit MSVC does not align things properly so we suppress high alignment annotations (#112480)
//@ ignore-i686-pc-windows-msvc
//@ ignore-i686-pc-windows-gnu

#![crate_type = "rlib"]

#[repr(align(16))]
pub struct S {
    arr: [u32; 4],
}

// CHECK-LABEL: @test1
// CHECK: store i32 0, ptr %{{.+}}, align 16
// CHECK: store i32 1, ptr %{{.+}}, align 4
// CHECK: store i32 2, ptr %{{.+}}, align 8
// CHECK: store i32 3, ptr %{{.+}}, align 4
#[no_mangle]
pub fn test1(s: &mut S) {
    s.arr[0] = 0;
    s.arr[1] = 1;
    s.arr[2] = 2;
    s.arr[3] = 3;
}

// CHECK-LABEL: @test2
// CHECK: store i32 4, ptr %{{.+}}, align 4
#[allow(unconditional_panic)]
#[no_mangle]
pub fn test2(s: &mut S) {
    s.arr[usize::MAX / 4 + 1] = 4;
}

// CHECK-LABEL: @test3
// CHECK: store i32 5, ptr %{{.+}}, align 4
#[no_mangle]
pub fn test3(s: &mut S, i: usize) {
    s.arr[i] = 5;
}

// CHECK-LABEL: @test4
// CHECK: store i32 6, ptr %{{.+}}, align 4
#[no_mangle]
pub fn test4(s: &mut S) {
    s.arr = [6; 4];
}
