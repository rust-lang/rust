#![feature(generators, negative_impls)]

fn assert_send<T: Send>(_: T) {}

struct S;
impl !Send for S {}

fn main() {
    println!("{}", std::mem::needs_drop::<S>());
    let g = || {
        let x = S; //~ type `S`
        yield; //~ `x` maybe used later
    };
    assert_send(g); //~ ERROR generator cannot be sent between threads
}
