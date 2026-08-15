fn main() {
    let mut my_var = false;
    let callback = || {
        my_var = true;
    };
    callback(); //~ ERROR E0596
}
