#![const_limit="1"]
//~^ ERROR the `#[const_limit]` attribute is an experimental feature [E0658]

const CONSTANT: usize = limit();

fn main() {}

const fn limit() -> usize {
    let x = 42;

    x * 42
}
