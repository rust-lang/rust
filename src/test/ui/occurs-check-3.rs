// error-pattern:mismatched types
// From Issue #778
enum clam<T> { a(T), }
fn main() { let c; c = clam::a(c); match c { clam::a::<isize>(_) => { } } }
