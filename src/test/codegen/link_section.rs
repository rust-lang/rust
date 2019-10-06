// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

// CHECK: @VAR1 = constant <{ [4 x i8] }> <{ [4 x i8] c"\01\00\00\00" }>, section ".test_one"
#[no_mangle]
#[link_section = ".test_one"]
#[cfg(target_endian = "little")]
pub static VAR1: u32 = 1;

#[no_mangle]
#[link_section = ".test_one"]
#[cfg(target_endian = "big")]
pub static VAR1: u32 = 0x01000000;

pub enum E {
    A(u32),
    B(f32)
}

// CHECK: @VAR2 = constant {{.*}}, section ".test_two"
#[no_mangle]
#[link_section = ".test_two"]
pub static VAR2: E = E::A(666);

// CHECK: @VAR3 = constant {{.*}}, section ".test_three"
#[no_mangle]
#[link_section = ".test_three"]
pub static VAR3: E = E::B(1.);

// CHECK: define void @fn1() {{.*}} section ".test_four" {
#[no_mangle]
#[link_section = ".test_four"]
pub fn fn1() {}
