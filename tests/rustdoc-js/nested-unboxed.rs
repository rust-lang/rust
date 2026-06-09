#![feature(rustdoc_internals)]

#[doc(search_unbox)]
pub struct Object<T, U>(T, U);

pub fn something() -> Result<Object<i32, u32>, bool> {
    loop {}
}
