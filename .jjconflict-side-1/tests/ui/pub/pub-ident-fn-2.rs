//@ run-rustfix

pub foo(_s: usize) { bar() }
//~^ ERROR missing `fn` for function definition

fn bar() {}

fn main() {
    foo(2);
}
