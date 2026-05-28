//@ ignore-wasm32 custom sections work differently on wasm
//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

// CHECK: @VAR1 = {{(dso_local )?}}constant [4 x i8] c"\01\00\00\00", section "__TEST,one"
#[no_mangle]
#[link_section = "__TEST,one"]
#[cfg(target_endian = "little")]
pub static VAR1: u32 = 1;

#[no_mangle]
#[link_section = "__TEST,one"]
#[cfg(target_endian = "big")]
pub static VAR1: u32 = 0x01000000;

pub enum E {
    A(u32),
    B(f32),
}

// CHECK: @VAR2 = {{(dso_local )?}}constant {{.*}}, section "__TEST,two"
#[no_mangle]
#[link_section = "__TEST,two"]
pub static VAR2: E = E::A(666);

// CHECK: @VAR3 = {{(dso_local )?}}constant {{.*}}, section "__TEST,three"
#[no_mangle]
#[link_section = "__TEST,three"]
pub static VAR3: E = E::B(1.);

// CHECK: define {{(dso_local )?}}void @fn1() {{.*}} section "__TEST,four" {
#[no_mangle]
#[link_section = "__TEST,four"]
pub fn fn1() {}
