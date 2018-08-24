fn main() {
    match Some("hi".to_string()) {
        Some(s) if s.len() == 0 => {},
        //~^ ERROR E0008
        _ => {},
    }
}
