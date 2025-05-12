fn main() {
    let mut my_var = false;
    let callback = move || {
        my_var = true;
    };
    callback(); //~ ERROR E0596
}
