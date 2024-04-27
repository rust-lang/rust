//@ check-pass
#![deny(unused_imports)]

struct S;

fn main() {
    use S;  //FIXME(unused_imports): ~ ERROR the item `S` is imported redundantly

    let _s = S;
}
