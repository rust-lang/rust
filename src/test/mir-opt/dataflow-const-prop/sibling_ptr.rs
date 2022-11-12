// unit-test: DataflowConstProp

// EMIT_MIR sibling_ptr.main.DataflowConstProp.diff
fn main() {
    let mut x: (u8, u8) = (0, 0);
    unsafe {
        let p = std::ptr::addr_of_mut!(x.0);
        *p.add(1) = 1;
    }
    let x1 = x.1;  // should not be propagated
}
