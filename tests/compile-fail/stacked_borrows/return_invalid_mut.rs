// Make sure that we cannot return a `&mut` that got already invalidated.
fn foo(x: &mut (i32, i32)) -> &mut i32 {
    let xraw = x as *mut (i32, i32);
    let ret = unsafe { &mut (*xraw).1 };
    let _val = *x; // invalidate xraw and its children
    ret //~ ERROR Mut reference with non-reactivatable tag Mut(Uniq
}

fn main() {
    foo(&mut (1, 2));
}
