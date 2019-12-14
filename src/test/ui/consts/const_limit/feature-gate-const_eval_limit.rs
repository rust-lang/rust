#![const_eval_limit="42"]
//~^ ERROR the `#[const_eval_limit]` attribute is an experimental feature [E0658]

const CONSTANT: usize = limit();

fn main() {
    assert_eq!(CONSTANT, 1764);
}

const fn limit() -> usize {
    let x = 42;

    x * 42
}
