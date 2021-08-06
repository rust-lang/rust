#![deny(unused_imports)]

struct S;

fn main() {
    use S;  //~ ERROR the item `S` is imported redundantly

    let _s = S;
}
