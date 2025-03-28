fn main() {
    let bar: isize = 5;
    fn foo() -> isize { return bar; } //~ ERROR can't capture dynamic environment in a fn item
}
