// An `is_some()` suggestion must apply to the whole expression, preserving its precedence.
//@ run-rustfix

fn foo(x: Option<i32>) {
    if true && x {}
    //~^ ERROR mismatched types
    //~| HELP use `Option::is_some` to test if the `Option` has a value

    let reference = &x;
    if *reference {}
    //~^ ERROR mismatched types
    //~| HELP use `Option::is_some` to test if the `Option` has a value

    if &x {}
    //~^ ERROR mismatched types
    //~| HELP use `Option::is_some` to test if the `Option` has a value

    if Value + Value {}
    //~^ ERROR mismatched types
    //~| HELP use `Option::is_some` to test if the `Option` has a value
}

struct Value;

impl std::ops::Add for Value {
    type Output = Option<i32>;

    fn add(self, _: Self) -> Self::Output {
        Some(1)
    }
}

fn main() {
    foo(Some(1));
}
