#![feature(existential_type)]

fn main() {}

// test that unused generic parameters are ok
existential type Two<T, U>: 'static;

fn one<T: 'static>(t: T) -> Two<T, T> {
    t
}

fn three<T: 'static, U: 'static>(_: T, u: U) -> Two<T, U> {
//~^ ERROR defining existential type use differs from previous
    u
}
