macro_rules! empty {
    () => { }
}

fn foo() -> bool { //~ ERROR mismatched
    { true } //~ ERROR mismatched
    empty!();
}

fn main() {}
