// Regression test for https://github.com/rust-lang/rust/issues/79033/
// rustc should be clear about the lifetime issue below where
// self is borrowed for the duration of the returned iterator
// but the lifetime of self will end when fn lines() returns.
//@ edition: 2024

#[derive(Debug, Clone, Copy)]
pub struct Location<'a> {
    pub filename: &'a str,
    pub start: LocationHalf,
    pub end: LocationHalf,
}

#[derive(Debug, Clone, Copy)]
pub struct LocationHalf {
    pub line: u32,
    pub column: u32,
}

impl Location<'_> {
    /// Returns an iterator over the line numbers and lines of this location.
    pub fn lines<'a>(self, source: &'a str) -> impl Iterator<Item = (u32, &'a str)> {
        let lines = source.split('\n');
        lines.enumerate().filter_map(|(i, line)| {
        //~^ ERROR closure may outlive the current function, but it borrows `self.start.line`, which is owned by the current function
        //~| ERROR closure may outlive the current function, but it borrows `self.end.line`, which is owned by the current function
            if self.start.line as usize <= i && i <= self.end.line as usize {
                Some((i as u32 + 1, line))
            } else {
                None
            }
        })
    }
}

fn main() {}
