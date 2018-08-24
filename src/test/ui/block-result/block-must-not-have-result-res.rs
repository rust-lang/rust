struct r;

impl Drop for r {
    fn drop(&mut self) {
        true //~  ERROR mismatched types
    }
}

fn main() {
}
