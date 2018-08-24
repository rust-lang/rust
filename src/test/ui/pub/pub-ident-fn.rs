// run-rustfix

pub   foo(_s: usize) -> bool { true }
//~^ ERROR missing `fn` for method definition

fn main() {
    foo(2);
}
