// Fixes issue where `+` in generics weren't parsed if they were part of a `+=`.

// compile-pass

struct Whitespace<T: Clone + = ()> { t: T }
struct TokenSplit<T: Clone +=  ()> { t: T }

fn main() {}
