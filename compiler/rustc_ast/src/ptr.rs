/// A pointer type that uniquely owns a heap allocation of type T.
///
/// This used to be its own type, but now it's just a typedef for `Box` and we are planning to
/// remove it soon.
pub type P<T> = Box<T>;

/// Construct a `P<T>` from a `T` value.
#[allow(non_snake_case)]
pub fn P<T>(value: T) -> P<T> {
    Box::new(value)
}
