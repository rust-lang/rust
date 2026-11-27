// Regression test for #162440. Diagnostics can run before closure capture
// analysis has inferred the tuple of upvar types. The capture-specific note
// must fall back instead of trying to access uninferred upvar types.

//@ compile-flags: -Znext-solver=globally

fn main() {
    Some([0]).map(|s| s[..]);
    //~^ ERROR the size for values of type `[{integer}]` cannot be known at compilation time
    //~| ERROR the size for values of type `[{integer}]` cannot be known at compilation time
    //~| ERROR the size for values of type `[{integer}]` cannot be known at compilation time
}
