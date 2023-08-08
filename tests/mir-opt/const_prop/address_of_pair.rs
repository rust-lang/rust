// unit-test: ConstProp

// EMIT_MIR address_of_pair.fn0.ConstProp.diff
pub fn fn0() -> bool {
    let mut pair = (1, false);
    let ptr = core::ptr::addr_of_mut!(pair.1);
    pair = (1, false);
    unsafe {
        *ptr = true;
    }
    let ret = !pair.1;
    return ret;
}

pub fn main() {
    println!("{}", fn0());
}
