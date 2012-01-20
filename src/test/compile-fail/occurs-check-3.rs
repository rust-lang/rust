// error-pattern:mismatched types
// From Issue #778
enum clam<T> { a(T); }
fn main() { let c; c = a(c); alt c { a::<int>(_) { } } }
