fn main() {}

trait T {
    fn qux() -> Option<usize> {
        let _ = if true {
        });
//~^ ERROR non-item in item list
//~| ERROR mismatched closing delimiter: `)`
//~| ERROR expected one of `.`, `;`
        Some(4)
    }
