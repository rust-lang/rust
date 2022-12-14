#[deprecated_safe(since = "TBD", note = "...")] //~ ERROR: the `#[deprecated_safe]` attribute is an experimental feature
unsafe fn deprecated_safe_fn() {}

#[deprecated_safe(since = "TBD", note = "...")] //~ ERROR: the `#[deprecated_safe]` attribute is an experimental feature
unsafe trait DeprecatedSafeTrait {}

fn main() {}
