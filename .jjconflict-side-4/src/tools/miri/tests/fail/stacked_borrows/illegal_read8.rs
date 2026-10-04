// Make sure that creating a raw ptr next to a shared ref works
// but the shared ref still gets invalidated when the raw ptr is used for writing.

fn main() {
    unsafe {
        use std::mem;
        let x = &mut 0;
        let y1: &i32 = mem::transmute(&*x); // launder lifetimes
        let y2 = x as *mut _;
        let _val = *y2;
        let _val = *y1;
        *y2 += 1;
        let _fail = *y1; //~ ERROR: /read access .* tag does not exist in the borrow stack/
    }
}
