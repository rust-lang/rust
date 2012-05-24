fn main() {
    let bar;
    fn baz(_x: int) { }
    bind baz(bar); //! ERROR use of possibly uninitialized variable: `bar`
}

