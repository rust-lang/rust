//! Check that small (less then 128bits on x86_64) homogeneous floats are either pass as an array
//! or by a pointer

// compile-flags: -C no-prepopulate-passes -O
// only-x86_64

#![crate_type = "lib"]

pub struct Foo {
    bar1: f32,
    bar2: f32,
    bar3: f32,
    bar4: f32,
}

// CHECK: define i64 @array_f32x2(i64 %0, i64 %1)
#[no_mangle]
pub fn array_f32x2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    todo!()
}

// CHECK: define void @array_f32x4([4 x float]* {{.*}} sret([4 x float]) {{.*}} %0, [4 x float]* {{.*}} %a, [4 x float]* {{.*}} %b)
#[no_mangle]
pub fn array_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    todo!()
}

// CHECK: define void @array_f32x4_nested(%Foo* {{.*}} sret(%Foo) {{.*}} %0, %Foo* {{.*}} %a, %Foo* {{.*}} %b)
#[no_mangle]
pub fn array_f32x4_nested(a: Foo, b: Foo) -> Foo {
    todo!()
}
