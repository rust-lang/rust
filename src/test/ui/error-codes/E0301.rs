fn main() {
    match Some(()) {
        None => { },
        option if option.take().is_none() => {}, //~ ERROR E0301
        Some(_) => { }
    }
}
