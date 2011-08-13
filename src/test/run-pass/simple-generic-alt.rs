

tag clam<T> { a(T); }

fn main() { let c = a(2); alt c { a::<int>(_) { } } }
