//@compile-flags: -Zmiri-recursive-validation

fn main() {
    let x = 3u8;
    let xref = &x;
    let xref_wrong_type: &bool = unsafe { std::mem::transmute(xref) }; //~ERROR: encountered 0x03, but expected a boolean
    let _val = *xref_wrong_type;
}
