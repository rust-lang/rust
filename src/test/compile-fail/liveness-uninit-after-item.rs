fn main() {
    let bar;
    fn baz(x: int) { }
    bind baz(bar); //! ERROR use of possibly uninitialized variable: `bar`
}

