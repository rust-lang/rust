//@ known-bug: #87577

#[derive(Debug)]
struct S<#[cfg(feature = "alloc")] N: A<T>> {}
