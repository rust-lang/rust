fn main() {
    let bar;
    fn baz(_x: isize) { }
    baz(bar); //~ ERROR E0381
}
