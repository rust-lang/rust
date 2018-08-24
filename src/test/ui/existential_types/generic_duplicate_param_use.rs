#![feature(existential_type)]

fn main() {}

existential type Two<T, U>: 'static; //~ ERROR type parameter `U` is unused

fn one<T: 'static>(t: T) -> Two<T, T> {
    t
}
