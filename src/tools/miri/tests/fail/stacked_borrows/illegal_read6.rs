// Creating a shared reference does not leak the data to raw pointers.
fn main() {
    unsafe {
        let x = &mut 0;
        let raw = x as *mut _;
        let x = &mut *x; // kill `raw`
        let _y = &*x; // this should not activate `raw` again
        let _val = *raw; //~ ERROR: /read access .* tag does not exist in the borrow stack/
    }
}
