fn main() {
    let bar;
    fn baz(_x: isize) { }
    baz(bar); //~ ERROR use of possibly uninitialized variable: `bar`
}
