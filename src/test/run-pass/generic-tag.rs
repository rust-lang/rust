

enum option<T> { some(@T), none, }

fn main() { let mut a: option<int> = some::<int>(@10); a = none::<int>; }
