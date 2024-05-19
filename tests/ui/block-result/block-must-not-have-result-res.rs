struct R;

impl Drop for R {
    fn drop(&mut self) {
        true //~  ERROR mismatched types
    }
}

fn main() {
}
