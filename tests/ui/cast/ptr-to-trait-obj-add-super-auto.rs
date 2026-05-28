//@ check-pass

trait Trait: Send {}
impl Trait for () {}

fn main() {
    // This is OK: `Trait` has `Send` super trait.
    &() as *const dyn Trait as *const (dyn Trait + Send);
}
