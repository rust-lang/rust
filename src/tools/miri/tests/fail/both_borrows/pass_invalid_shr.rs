//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

// Make sure that we cannot pass by argument a `&` that got already invalidated.
fn foo(_: &i32) {}

fn main() {
    let x = &mut 42;
    let xraw = x as *mut _;
    let xref = unsafe { &*xraw };
    unsafe { *xraw = 42 }; // unfreeze
    foo(xref);
    //~[stack]^ ERROR: /retag .* tag does not exist in the borrow stack/
    //~[tree]| ERROR: /reborrow through .* is forbidden/
}
