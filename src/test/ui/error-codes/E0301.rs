fn main() {
    match Some(()) {
        None => { },
        option if option.take().is_none() => {},
        Some(_) => { } //~^ ERROR E0596
    }
}
