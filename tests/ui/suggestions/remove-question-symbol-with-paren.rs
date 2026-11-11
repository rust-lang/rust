// https://github.com/rust-lang/rust/issues/114392

fn foo() -> Option<()> {
    let x = Some(());
    (x?)
    //~^ ERROR `?` operator has incompatible types
}

fn main() {}
