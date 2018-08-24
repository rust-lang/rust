// pretty-expanded FIXME #23616

enum clam<T> { a(T), }

pub fn main() { let c = clam::a(2); match c { clam::a::<isize>(_) => { } } }
