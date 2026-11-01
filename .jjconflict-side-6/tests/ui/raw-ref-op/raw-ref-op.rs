//@ run-pass

fn main() {
    let mut x = 123;
    let c_p = &raw const x;
    let m_p = &raw mut x;
    let i_r = &x;
    assert!(c_p == i_r);
    assert!(c_p == m_p);
    unsafe { assert!(*c_p == *i_r ); }
}
