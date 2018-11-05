// Make sure that we cannot pass by argument a `&` that got already invalidated.
fn foo(_: &i32) {}

fn main() {
    let x = &mut 42;
    let xraw = &*x as *const _;
    let xref = unsafe { &*xraw };
    *x = 42; // invalidate xraw
    foo(xref); //~ ERROR does not exist on the stack
}
