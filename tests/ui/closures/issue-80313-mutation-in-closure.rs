//@ check-pass
fn main() {
    let mut my_var = false;
    let callback = || {
        my_var = true;
    };
    callback(); //~ WARNING E0596
}
