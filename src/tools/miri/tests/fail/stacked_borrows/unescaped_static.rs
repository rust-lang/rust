static ARRAY: [u8; 2] = [0, 1];

fn main() {
    let ptr_to_first = &ARRAY[0] as *const u8;
    // Illegally use this to access the 2nd element.
    let _val = unsafe { *ptr_to_first.add(1) }; //~ ERROR: /read access .* tag does not exist in the borrow stack/
}
