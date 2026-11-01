//@ revisions: edition2024 polonius_alpha
//@ ignore-compare-mode-polonius (explicit revisions)
//@ check-pass

// Tests that liveness for regions in associated types considers outlives bounds.


trait Updater {
    // Because `'b` isn't known to outlive `'a`, then we know that
    // `Self::Changes<'a, 'b>` should not need to consider `'b` to be live.
    type Changes<'a, 'b>: 'a
    where
        Self: 'a;
    fn changes<'a, 'b>(&'a self, _value: &'b u8) -> Self::Changes<'a, 'b>;
    fn run(&self) {
        let mut cluster = 0u8;
        let changes = self.changes(&cluster);
        cluster = 1;
        let _ = changes;
    }
}

fn main() {}
