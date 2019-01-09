fn main() {
    match Some(()) {
        None => { },
        option if { option = None; false } => { }, //~ ERROR E0302
        Some(_) => { }
    }
}
