fn main() {
    let mut my_var = false;
    let callback = || {
        &mut my_var;
    };
    callback(); //~ ERROR E0596
}
