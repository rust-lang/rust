// Validation detects that we are casting & to &mut and so it changes why we fail
// compile-flags: -Zmiri-disable-validation

static X: usize = 5;

#[allow(mutable_transmutes)]
fn main() {
    unsafe {
        *std::mem::transmute::<&usize, &mut usize>(&X) = 6; //~ ERROR tried to modify constant memory
        assert_eq!(X, 6);
    }
}
