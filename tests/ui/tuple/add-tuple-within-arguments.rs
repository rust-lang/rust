fn foo(s: &str, a: (i32, i32), s2: &str) {}

fn bar(s: &str, a: (&str,), s2: &str) {}

fn main() {
    foo("hi", 1, 2, "hi");
    //~^ ERROR function takes 3 arguments but 4 arguments were supplied
    bar("hi", "hi", "hi");
    //~^ ERROR mismatched types
}
