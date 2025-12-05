// Expansion drives parsing, so conditional compilation will strip
// out outline modules and we will never attempt parsing them.

//@ check-pass

fn main() {}

#[cfg(false)]
mod foo {
    mod bar {
        mod baz; // This was an error before.
    }
}
