//@ edition: 2024
trait Demo {}

impl dyn Demo {
    pub fn report(&self) -> u32 {
        let sum = |a: u32,
                   b: u32,
                   c: u32| {
            a + b + c
        };
        sum(1, 2, 3)
    }

    fn check(&self, val: Option<u32>, num: Option<u32>) {
        if let Some(b) = val
        && let Some(c) = num {
        && b == c {
        }
    }
}

fn main() { } //~ ERROR this file contains an unclosed delimiter
