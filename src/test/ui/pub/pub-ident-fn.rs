// run-rustfix

pub   foo(_s: usize) -> bool { true }
//~^ ERROR missing `fn` for function definition

fn main() {
    foo(2);
}
