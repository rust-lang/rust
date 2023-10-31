//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

// Make sure that we cannot pass by argument a `&` that got already invalidated.
fn foo(_: (&i32, &i32)) {}

fn main() {
    let x = &mut (42i32, 31i32);
    let xraw0 = &mut x.0 as *mut _;
    let xraw1 = &mut x.1 as *mut _;
    let pair_xref = unsafe { (&*xraw0, &*xraw1) };
    unsafe { *xraw0 = 42 }; // unfreeze
    foo(pair_xref);
    //~[stack]^ ERROR: /retag .* tag does not exist in the borrow stack/
    //~[tree]| ERROR: /reborrow through .* is forbidden/
}
