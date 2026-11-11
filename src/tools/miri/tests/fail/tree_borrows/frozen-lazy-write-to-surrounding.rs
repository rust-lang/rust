//@compile-flags: -Zmiri-tree-borrows

fn main() {
    // Since the "inside" part is `!Freeze`, the permission to mutate is gone.
    let pair = ((), 1);
    let x = &pair.0;
    let ptr = (&raw const *x).cast::<i32>().cast_mut();
    unsafe { ptr.write(0) }; //~ERROR: /write access .* forbidden/
}
