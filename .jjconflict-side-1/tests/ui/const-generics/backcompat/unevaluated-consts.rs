//@ check-pass

// If we allow the parent generics here without using lazy normalization
// this results in a cycle error.
struct Foo<T, U>(T, U);

impl<T> From<[u8; 1 + 1]> for Foo<T, [u8; 1 + 1]> {
    fn from(value: [u8; 1 + 1]) -> Foo<T, [u8; 1 + 1]> {
        todo!();
    }
}

fn break_me<T>()
where
    [u8; 1 + 1]: From<[u8; 1 + 1]>
{}

fn main() {}
