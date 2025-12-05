// Using a raw invalidates derived `&mut` even for reading.
fn main() {
    let mut x = 2;
    let xref1 = &mut x;
    let xraw = xref1 as *mut _;
    let xref2 = unsafe { &mut *xraw };
    let _val = unsafe { *xraw }; // use the raw again, this invalidates xref2 *even* with the special read except for uniq refs
    let _illegal = *xref2; //~ ERROR: /read access .* tag does not exist in the borrow stack/
}
