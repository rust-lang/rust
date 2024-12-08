macro_rules! foo {
    ($e: expr) => { 'x: loop { $e } }
}

pub fn main() {
    foo!(break 'x); //~ ERROR use of undeclared label `'x`
}
