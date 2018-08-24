pub foo(s: usize) { bar() }
//~^ ERROR missing `fn` for method definition

fn main() {
    foo(2);
}
