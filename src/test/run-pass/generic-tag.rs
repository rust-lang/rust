

enum option<T> { some(@T), none, }

fn main() { let a: option<int> = some::<int>(@10); a = none::<int>; }
