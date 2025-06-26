// NOTE: this used to panic in debug builds (by a sanity assertion)
// and not emit any lint on release builds. See https://github.com/rust-lang/rust/issues/142891.
#![inline = ""]
//~^ ERROR valid forms for the attribute are `#[inline(always|never)]` and `#[inline]`
//~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

fn main() {}
