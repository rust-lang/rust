fn main() {
    match Some(()) {
        None => { },
        option if { option = None; false } => { }, //~ ERROR E0302
        //~^ ERROR cannot assign to `option`, as it is immutable for the pattern guard
        Some(_) => { }
    }
}
