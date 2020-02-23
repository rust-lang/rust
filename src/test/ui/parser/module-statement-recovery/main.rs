fn main() {}

fn foo() {}

let x = 0; //~ ERROR statements cannot reside in modules
x;
x;
x;
x;
x;
x;
foo()?;
Ok(42u16)

struct X;

if true {} //~ ERROR statements cannot reside in modules
