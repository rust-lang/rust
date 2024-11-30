//@ check-pass
fn main() {
    let mut my_var = false;
    let callback = || {
        &mut my_var;
    };
    callback(); //~ WARNING E0596
}
