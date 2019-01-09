#![warn(clippy::wrong_self_convention)]
#![warn(clippy::wrong_pub_self_convention)]
#![allow(dead_code, clippy::trivially_copy_pass_by_ref)]

fn main() {}

#[derive(Clone, Copy)]
struct Foo;

impl Foo {
    fn as_i32(self) {}
    fn as_u32(&self) {}
    fn into_i32(self) {}
    fn is_i32(self) {}
    fn is_u32(&self) {}
    fn to_i32(self) {}
    fn from_i32(self) {}

    pub fn as_i64(self) {}
    pub fn into_i64(self) {}
    pub fn is_i64(self) {}
    pub fn to_i64(self) {}
    pub fn from_i64(self) {}
    // check whether the lint can be allowed at the function level
    #[allow(clippy::wrong_self_convention)]
    pub fn from_cake(self) {}

    fn as_x<F: AsRef<Self>>(_: F) {}
    fn as_y<F: AsRef<Foo>>(_: F) {}
}

struct Bar;

impl Bar {
    fn as_i32(self) {}
    fn as_u32(&self) {}
    fn into_i32(&self) {}
    fn into_u32(self) {}
    fn is_i32(self) {}
    fn is_u32(&self) {}
    fn to_i32(self) {}
    fn to_u32(&self) {}
    fn from_i32(self) {}

    pub fn as_i64(self) {}
    pub fn into_i64(&self) {}
    pub fn is_i64(self) {}
    pub fn to_i64(self) {}
    pub fn from_i64(self) {}

    // test for false positives
    fn as_(self) {}
    fn into_(&self) {}
    fn is_(self) {}
    fn to_(self) {}
    fn from_(self) {}
    fn to_mut(&mut self) {}
}
