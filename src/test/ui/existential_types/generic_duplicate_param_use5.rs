#![feature(existential_type)]

fn main() {}

// test that unused generic parameters are ok
existential type Two<T, U>: 'static;

fn one<T: 'static>(t: T) -> Two<T, T> {
    t
}

fn two<T: 'static, U: 'static>(t: T, _: U) -> Two<U, T> {
//~^ ERROR defining existential type use differs from previous
    t
}
