#![deny(unused_imports)]

struct S;

fn main() {
    use S; //~ ERROR redundant import

    let _s = S;
}
