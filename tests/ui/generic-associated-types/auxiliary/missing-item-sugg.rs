pub trait Foo {
    type Gat<T>
    where
        T: std::fmt::Display;
}
