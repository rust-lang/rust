pub struct S<T>(Option<T>);
pub fn mk<T>() -> S<T> { S(None) }
