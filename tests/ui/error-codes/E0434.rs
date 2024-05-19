fn foo() {
    let y = 5;
    fn bar() -> u32 {
        y //~ ERROR E0434
    }
}

fn main () {
}
