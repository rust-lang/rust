#![no_std]

// @has "$.index[*][?(@.name=='FooUnsafe')]"
// @is "$.index[*][?(@.name=='FooUnsafe')].inner.trait.is_object_safe" false
pub trait FooUnsafe {
    fn foo() -> Self;
}

// @has "$.index[*][?(@.name=='BarUnsafe')]"
// @is "$.index[*][?(@.name=='BarUnsafe')].inner.trait.is_object_safe" false
pub trait BarUnsafe<T> {
    fn foo(i: T);
}

// @has "$.index[*][?(@.name=='FooSafe')]"
// @is "$.index[*][?(@.name=='FooSafe')].inner.trait.is_object_safe" true
pub trait FooSafe {
    fn foo(&self);
}
