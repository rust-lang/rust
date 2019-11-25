// FIXME(#64490): make this run-pass

#![feature(raw_ref_op)]

fn main() {
    let mut x = 123;
    let c_p = &raw const x;                     //~ ERROR not yet implemented
    let m_p = &raw mut x;                       //~ ERROR not yet implemented
    let i_r = &x;
    assert!(c_p == i_r);
    assert!(c_p == m_p);
    unsafe { assert!(*c_p == *i_r ); }
}
