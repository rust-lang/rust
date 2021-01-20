extern { fn foo(); }

pub fn main() {
    unsafe { foo(); }
    assert_eq!(7f32.powi(3), 343f32);
}
