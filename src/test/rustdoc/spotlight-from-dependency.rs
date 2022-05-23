#![crate_name = "foo"]

use std::iter::Iterator;

// @has foo/struct.Odd.html
// @has - '//*[@id="method.new"]//span[@class="notable-traits"]//code/span' 'impl Iterator for Odd'
pub struct Odd {
    current: usize,
}

impl Odd {
    pub fn new() -> Odd {
        Odd { current: 1 }
    }
}

impl Iterator for Odd {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.current += 2;
        Some(self.current - 2)
    }
}
