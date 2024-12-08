//@ run-rustfix

enum VecOrMap {
    //~^ HELP: perhaps you meant to use `struct` here
    vec: Vec<usize>,
    //~^ ERROR expected one of `(`, `,`, `=`, `{`, or `}`, found `:`
    //~| HELP: enum variants can be `Variant`, `Variant = <integer>`, `Variant(Type, ..., TypeN)` or `Variant { fields: Types }`
}

fn main() {
    let o = VecOrMap { vec: vec![1, 2, 3] };
    println!("{:?}", o.vec);
}
