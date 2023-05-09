
const UNALIGNED_READ: () = unsafe {
    let x = &[0u8; 4];
    let ptr = x.as_ptr().cast::<u32>();
    ptr.read(); //~ERROR: evaluation of constant value failed
};

fn main() {
    let _x = UNALIGNED_READ;
}
