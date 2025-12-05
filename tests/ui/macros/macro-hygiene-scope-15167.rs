//! Regression test for https://github.com/rust-lang/rust/issues/15167

// macro f should not be able to inject a reference to 'n'.

macro_rules! f { () => (n) }
//~^ ERROR cannot find value `n` in this scope
//~| ERROR cannot find value `n` in this scope
//~| ERROR cannot find value `n` in this scope
//~| ERROR cannot find value `n` in this scope

fn main() -> (){
    for n in 0..1 {
        println!("{}", f!());
    }

    if let Some(n) = None {
        println!("{}", f!());
    }

    if false {
    } else if let Some(n) = None {
        println!("{}", f!());
    }

    while let Some(n) = None {
        println!("{}", f!());
    }
}
