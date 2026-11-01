// This is a reduction of a concrete test illustrating a case that was
// annoying to Rust developer niconii (see comment thread on #21114).
//
// With resolving issue #54556, pnkfelix hopes that the new diagnostic
// output produced by NLL helps to *explain* the semantic significance
// of temp drop order, and thus why inserting a semi-colon after the
// `if let` expression in `main` works.

//@ revisions: edition2021 edition2024
//@ [edition2021] edition: 2021
//@ [edition2024] edition: 2024
//@ [edition2024] check-pass

struct Mutex;
struct MutexGuard<'a>(&'a Mutex);

impl Drop for Mutex { fn drop(&mut self) { println!("Mutex::drop"); } }
impl<'a> Drop for MutexGuard<'a> { fn drop(&mut self) { println!("MutexGuard::drop");  } }

impl Mutex {
    fn lock(&self) -> Result<MutexGuard<'_>, ()> { Ok(MutexGuard(self)) }
}

fn main() {
    let counter = Mutex;

    if let Ok(_) = counter.lock() { }
    //[edition2021]~^ ERROR: does not live long enough

    // Up until Edition 2021:
    // With this code as written, the dynamic semantics here implies
    // that `Mutex::drop` for `counter` runs *before*
    // `MutexGuard::drop`, which would be unsound since `MutexGuard`
    // still has a reference to `counter`.
    //
    // The goal of #54556 is to explain that within a compiler
    // diagnostic.

    // From Edition 2024:
    // Now `MutexGuard::drop` runs *before* `Mutex::drop` because
    // the lifetime of the `MutexGuard` is shortened to cover only
    // from `if let` until the end of the consequent block.
    // Therefore, Niconii's issue is properly solved thanks to the new
    // temporary lifetime rule for `if let`s.
}
