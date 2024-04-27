fn main() {
    match true {
        true => 'a: {
            break 'a;
        }
        _ => (),
    }
}
