// Regression test for #153391.

//@ edition:2024
//@ ignore-parallel-frontend query cycle

trait A {
    fn g() -> B;
    //~^ ERROR expected a type, found a trait
}

trait B {
    fn bar(&self, x: &A);
    //~^ ERROR expected a type, found a trait
}

fn main() {}
