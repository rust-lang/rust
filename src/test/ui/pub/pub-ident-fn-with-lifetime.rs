pub   foo<'a>(_s: &'a usize) -> bool { true }
//~^ ERROR missing `fn` for method definition

fn main() {
    foo(2);
}
