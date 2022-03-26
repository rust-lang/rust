#![feature(used_on_fn_def)]

#[used]
static FOO: u32 = 0; // OK

#[used]
fn foo() {} // OK

#[used] //~ ERROR attribute must be applied to a `static` variable or a function definition
struct Foo {}

#[used] //~ ERROR attribute must be applied to a `static` variable or a function definition
trait Bar {}

#[used] //~ ERROR attribute must be applied to a `static` variable or a function definition
impl Bar for Foo {}

fn main() {}
