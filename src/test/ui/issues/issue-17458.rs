static X: usize = unsafe { 0 as *const usize as usize };
//~^ ERROR: casting pointers to integers in statics is unstable

fn main() {
    assert_eq!(X, 0);
}
