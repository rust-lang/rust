fn main() {
    let ref is_ref @ is_val = 42;
    *is_ref;
    *is_val;
    //~^ ERROR cannot be dereferenced

    let is_val @ ref is_ref = 42;
    *is_ref;
    *is_val;
    //~^ ERROR cannot be dereferenced
}
