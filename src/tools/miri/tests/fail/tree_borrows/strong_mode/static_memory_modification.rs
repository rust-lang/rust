// Tests that inserting an implicit write to a read-only allocation under the strong mode generates the correct error message
//@compile-flags: -Zmiri-tree-borrows
static X: usize = 5;

#[allow(mutable_transmutes)]
fn main() {
    let x = unsafe { std::mem::transmute::<&usize, &mut usize>(&X) };
    foo(x);
}

fn foo(_x: &mut usize) {} //~ ERROR: writing to alloc1 which is read-only
