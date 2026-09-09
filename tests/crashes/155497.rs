//@ known-bug: #155497
//@ compile-flags: -Wrust-2021-incompatible-closure-captures
struct Foo((u32, i128));

fn main() {
    type T = impl async FnOnce() -> T;
    let foo: T = Foo();
    let x = move || {
        let x = move || {
            let Foo((a, b)) = foo;
        };
    };
}
