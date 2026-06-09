//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

// Make sure that we cannot return a `&` that got already invalidated, not even in an `Option`.
fn foo(x: &mut (i32, i32)) -> Option<&i32> {
    let xraw = x as *mut (i32, i32);
    let ret = Some(unsafe { &(*xraw).1 });
    unsafe { *xraw = (42, 23) }; // unfreeze
    ret
    //~[stack]^ ERROR: /retag .* tag does not exist in the borrow stack/
    //~[tree]| ERROR: /reborrow through .* is forbidden/
}

fn main() {
    match foo(&mut (1, 2)) {
        Some(_x) => {}
        None => {}
    }
}
