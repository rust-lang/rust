

enum clam<T> { a(T), }

fn main() { let c = a(2); match c { a::<int>(_) => { } } }
