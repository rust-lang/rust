// error-pattern:mismatched types
// From Issue #778
tag clam<T> { a(T); }
fn main() { let c; c = a(c); alt c { a[int](_) { } } }
