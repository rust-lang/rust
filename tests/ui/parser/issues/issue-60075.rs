fn main() {}

trait T {
    fn qux() -> Option<usize> {
        let _ = if true {
        }); //~ ERROR mismatched closing delimiter
        Some(4)
    }
