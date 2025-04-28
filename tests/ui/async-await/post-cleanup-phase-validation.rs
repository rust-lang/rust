//@ compile-flags: -Zvalidate-mir
//@ edition: 2024
//@ build-pass

// Regression test that we don't ICE when encountering a transmute in a coroutine's
// drop shim body, which is conceptually in the Runtime phase but wasn't having the
// phase updated b/c the pass manager neither optimizes nor updates the phase for
// drop shim bodies.

struct HasDrop;
impl Drop for HasDrop {
    fn drop(&mut self) {}
}

fn main() {
    async {
        vec![async { HasDrop }.await];
    };
}
