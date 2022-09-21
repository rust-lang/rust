// Make sure that we cannot return a `&mut` that got already invalidated, not even in an `Option`.
// Due to shallow reborrowing, the error only surfaces when we look into the `Option`.
fn foo(x: &mut (i32, i32)) -> Option<&mut i32> {
    let xraw = x as *mut (i32, i32);
    let ret = unsafe { &mut (*xraw).1 }; // let-bind to avoid 2phase
    let ret = Some(ret);
    let _val = unsafe { *xraw }; // invalidate xref
    ret
}

fn main() {
    match foo(&mut (1, 2)) {
        Some(_x) => {} //~ ERROR: /retag .* tag does not exist in the borrow stack/
        None => {}
    }
}
