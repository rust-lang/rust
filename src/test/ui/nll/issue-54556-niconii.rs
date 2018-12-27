// This is a reduction of a concrete test illustrating a case that was
// annoying to Rust developer niconii (see comment thread on #21114).
//
// With resolving issue #54556, pnkfelix hopes that the new diagnostic
// output produced by NLL helps to *explain* the semantic significance
// of temp drop order, and thus why inserting a semi-colon after the
// `if let` expression in `main` works.

struct Mutex;
struct MutexGuard<'a>(&'a Mutex);

impl Drop for Mutex { fn drop(&mut self) { println!("Mutex::drop"); } }
impl<'a> Drop for MutexGuard<'a> { fn drop(&mut self) { println!("MutexGuard::drop");  } }

impl Mutex {
    fn lock(&self) -> Result<MutexGuard, ()> { Ok(MutexGuard(self)) }
}

fn main() {
    let counter = Mutex;

    if let Ok(_) = counter.lock() { } //~ ERROR does not live long enough

    // With this code as written, the dynamic semantics here implies
    // that `Mutex::drop` for `counter` runs *before*
    // `MutexGuard::drop`, which would be unsound since `MutexGuard`
    // still has a reference to `counter`.
    //
    // The goal of #54556 is to explain that within a compiler
    // diagnostic.
}
