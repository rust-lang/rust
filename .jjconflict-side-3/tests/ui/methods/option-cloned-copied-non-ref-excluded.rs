// Tests that the guard for `.cloned()`/`.copied()` on `Option<T>` correctly
// excludes inner types where the targeted diagnostic would be wrong:
//
// - `Option<&T>`: `Option<&T>` has inherent `cloned()`/`copied()` methods,
//   so these compile successfully and the guard must not fire.
// - `Option<T>` (generic param): falls through to the standard "not an
//   iterator / call .into_iter() first" diagnostic.
//
// See https://github.com/rust-lang/rust/issues/151147

// Reference inner type: these should compile without error.
pub fn cloned_on_ref(x: Option<&i32>) -> Option<i32> {
    x.cloned()
}

pub fn copied_on_ref(x: Option<&i32>) -> Option<i32> {
    x.copied()
}

// Generic param inner type: falls through to the standard diagnostic.
pub fn cloned_on_param<T: Clone>(x: Option<T>) {
    x.cloned();
    //~^ ERROR no method named `cloned` found for enum `Option<T>` in the current scope
    //~| HELP call `.into_iter()` first
}

fn main() {}
