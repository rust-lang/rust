//@ revisions: edition2024 polonius_alpha
//@ ignore-compare-mode-polonius (explicit revisions)
//@ edition: 2024
//@ [edition2024] compile-flags: -Zpolonius=nll
//@ [polonius_alpha] compile-flags: -Zpolonius=next

// Tests that liveness for regions in associated types considers outlives
// bounds, and the transitive implied outlives bounds from those.

trait Updater {
    // Because `'b` is known to outlive `'a`, then we must consider that `'b`
    // may be live in `Self::Changes<'a, 'b>`.
    type Changes<'a, 'b: 'a>: 'a
    where
        Self: 'a;
    fn changes<'a, 'b>(&'a self, _value: &'b u8) -> Self::Changes<'a, 'b>;
    fn run(&self) {
        let mut cluster = 0u8;
        let changes = self.changes(&cluster);
        cluster = 1; //[edition2024,polonius_alpha]~ ERROR cannot assign to `cluster` because it is borrowed
        let _ = changes;
    }
}

fn main() {}
