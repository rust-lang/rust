// error-pattern: illegal recursive enum type

enum list<T> { cons(T, list<T>), nil }

fn main() {}
