fn main() {
    match 3 {
        t if match t {
            _ => true,
        } => {}
        _ => {}
    }
}
