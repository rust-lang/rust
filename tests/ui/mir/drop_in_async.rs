//@ run-pass
//@ compile-flags: --edition=2024 -Zvalidate-mir

// async drops are elaborated earlier than non-async ones.
// See <https://github.com/rust-lang/rust/issues/137243>

struct HasDrop;
impl Drop for HasDrop {
    fn drop(&mut self) {}
}

fn main() {
    let _ = async {
        vec![async { HasDrop }.await];
    };
}
