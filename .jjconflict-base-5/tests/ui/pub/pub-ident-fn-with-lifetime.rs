//@ run-rustfix

pub foo<'a>(_s: &'a usize) -> bool { true }
//~^ ERROR missing `fn` for function definition

fn main() {
    foo(&2);
}
