//@ check-pass
fn main() {
    let mut my_var = false;
    let callback = move || {
        &mut my_var;
    };
    callback(); //~ WARNING E0596
}
