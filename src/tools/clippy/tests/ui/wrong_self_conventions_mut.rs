#![warn(clippy::wrong_self_convention)]
#![allow(dead_code)]

fn main() {}

mod issue6758 {
    pub enum Test<T> {
        One(T),
        Many(Vec<T>),
    }

    impl<T> Test<T> {
        // If a method starts with `to_` and not ends with `_mut` it should expect `&self`
        pub fn to_many(&mut self) -> Option<&mut [T]> {
            match self {
                Self::Many(data) => Some(data),
                _ => None,
            }
        }

        // If a method starts with `to_` and ends with `_mut` it should expect `&mut self`
        pub fn to_many_mut(&self) -> Option<&[T]> {
            match self {
                Self::Many(data) => Some(data),
                _ => None,
            }
        }
    }
}
