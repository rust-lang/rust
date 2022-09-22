// Make sure that we cannot return a `&mut` that got already invalidated, not even in a tuple.
// Due to shallow reborrowing, the error only surfaces when we look into the tuple.
fn foo(x: &mut (i32, i32)) -> (&mut i32,) {
    let xraw = x as *mut (i32, i32);
    let ret = (unsafe { &mut (*xraw).1 },);
    let _val = unsafe { *xraw }; // invalidate xref
    ret
}

fn main() {
    foo(&mut (1, 2)).0; //~ ERROR: /retag .* tag does not exist in the borrow stack/
}
