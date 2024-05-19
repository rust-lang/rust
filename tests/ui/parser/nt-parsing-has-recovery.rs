macro_rules! foo {
    ($e:expr) => {}
}

foo!(1 + @); //~ ERROR expected expression, found `@`
foo!(1 + @); //~ ERROR expected expression, found `@`

fn main() {
    let _recovery_witness: () = 0; //~ ERROR mismatched types
}
