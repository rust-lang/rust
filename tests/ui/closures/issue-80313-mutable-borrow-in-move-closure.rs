fn main() {
    let mut my_var = false;
    let callback = move || {
        &mut my_var;
    };
    callback(); //~ ERROR E0596
}
