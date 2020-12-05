static X: usize = unsafe { core::ptr::null::<usize>() as usize };
//~^ ERROR: casting pointers to integers in statics is unstable

fn main() {
    assert_eq!(X, 0);
}
