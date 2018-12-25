pub foo(s: usize) { bar() }
//~^ ERROR missing `fn` for function definition

fn main() {
    foo(2);
}
