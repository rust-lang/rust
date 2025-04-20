macro_rules! foo {
    () => { break 'x; } //~ ERROR use of undeclared label `'x`
}

pub fn main() {
    'x: for _ in 0..1 {
        foo!();
    };
}
