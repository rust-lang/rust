// error-pattern: can't capture dynamic environment in a fn item

fn main() {
    let bar: isize = 5;
    fn foo() -> isize { return bar; }
}
