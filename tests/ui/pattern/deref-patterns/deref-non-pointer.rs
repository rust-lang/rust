fn main() {
    match *1 { //~ ERROR: cannot be dereferenced
        _ => {}
    }
}
