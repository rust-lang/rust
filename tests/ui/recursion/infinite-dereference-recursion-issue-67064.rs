//@ compile-flags: -Zwrite-long-types-to-disk=yes

// Regression test for: https://github.com/rust-lang/rust/issues/67064
// This used to take a long time to compile because dereferencing led to
// recursive expansion of the type of the struct.

struct S<T>(T);

impl<T> std::ops::Deref for S<T> {
    type Target = S<(T, T)>;
    fn deref(&self) -> &Self::Target {
        self.thing()
        //~^ ERROR: reached the recursion limit while auto-dereferencing
        //~| ERROR: no method named `thing` found for reference
    }
}

fn main() {}
