// This attempts to modify `x.1` via a pointer derived from `addr_of_mut!(x.0)`.
// According to Miri, that is UB. However, T-opsem has not finalized that
// decision and as such we cannot rely on it in optimizations. Consequently,
// DataflowConstProp must treat the `addr_of_mut!(x.0)` as potentially being
// used to modify `x.1` - if it did not, then it might incorrectly assume that it
// can infer the value of `x.1` at the end of this function.

// unit-test: DataflowConstProp

// EMIT_MIR sibling_ptr.main.DataflowConstProp.diff
fn main() {
    let mut x: (u8, u8) = (0, 0);
    unsafe {
        let p = std::ptr::addr_of_mut!(x.0);
        *p.add(1) = 1;
    }
    let x1 = x.1; // should not be propagated
}
