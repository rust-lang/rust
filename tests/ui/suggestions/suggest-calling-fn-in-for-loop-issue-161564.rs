//@ edition: 2021

//! Check that a fn item used as the iterator of a `for` loop is suggested to be called, the way
//! it already is when it is passed as a function argument.

struct S;

trait T {
    fn assoc_in_trait() -> std::vec::IntoIter<u8>;
}

impl T for S {
    fn assoc_in_trait() -> std::vec::IntoIter<u8> {
        vec![1u8].into_iter()
    }
}

impl S {
    fn inherent_assoc() -> impl Iterator<Item = u8> {
        [1u8].into_iter()
    }
}

fn free_fn() -> impl Iterator<Item = u8> {
    [1u8].into_iter()
}

fn main() {
    for _ in S::inherent_assoc {} //~ ERROR [E0277]
    for _ in <S as T>::assoc_in_trait {} //~ ERROR [E0277]
    for _ in free_fn {} //~ ERROR [E0277]

    let closure = || vec![1u8].into_iter();
    for _ in closure {} //~ ERROR [E0277]

    for _ in || vec![1u8].into_iter() {} //~ ERROR [E0277]
}
