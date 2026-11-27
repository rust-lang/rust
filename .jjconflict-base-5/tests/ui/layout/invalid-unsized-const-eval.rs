// issue: #124182

//! This test used to trip an assertion in const eval, because `layout_of(LazyLock)`
//! returned `Ok` with an unsized layout when a sized layout was expected.
//! It was fixed by making `layout_of` always return `Err` for types that
//! contain unsized fields in unexpected locations.

struct LazyLock {
    data: (dyn Sync, ()), //~ ERROR the size for values of type
}

static EMPTY_SET: LazyLock = todo!();
//~^ ERROR the type `(dyn Sync, ())` has an unknown layout

fn main() {}
