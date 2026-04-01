pub trait SomeTrait {}

impl SomeTrait for () {}

// Adding this `impl` would cause errors in this crate's dependent,
// so it would be a breaking change. We explicitly don't add this impl,
// as the dependent crate already assumes this impl exists and thus already
// does not compile.
//impl SomeTrait for i32 {}
