// Make sure that we cannot return a `&mut` that got already invalidated, not even in an `Option`.
fn foo(x: &mut (i32, i32)) -> Option<&mut i32> {
    let xraw = x as *mut (i32, i32);
    let ret = Some(unsafe { &mut (*xraw).1 });
    let _val = unsafe { *xraw }; // invalidate xref
    ret //~ ERROR does not exist on the borrow stack
}

fn main() {
    foo(&mut (1, 2));
}
