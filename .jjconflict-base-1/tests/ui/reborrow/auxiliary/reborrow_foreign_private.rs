#![allow(dead_code)]

#[derive(Clone, Copy)]
pub struct ForeignRef<'a> {
    value: &'a i32,
}
