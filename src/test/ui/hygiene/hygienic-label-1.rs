macro_rules! foo {
    () => { break 'x; } //~ ERROR use of undeclared label `'x`
}

pub fn main() {
    'x: loop { foo!() }
}
