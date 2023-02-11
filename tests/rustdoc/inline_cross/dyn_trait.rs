#![crate_name = "user"]

// aux-crate:dyn_trait=dyn_trait.rs
// edition:2021

// @has user/type.Ty0.html
// @has - '//*[@class="rust item-decl"]//code' "dyn for<'any> FnOnce(&'any str) -> bool + 'static"
// FIXME(fmease): Hide default lifetime bound `'static`
pub use dyn_trait::Ty0;

// @has user/type.Ty1.html
// @has - '//*[@class="rust item-decl"]//code' "dyn Display + 'obj"
pub use dyn_trait::Ty1;

// @has user/type.Ty2.html
// @has - '//*[@class="rust item-decl"]//code' "dyn for<'a, 'r> Container<'r, Item<'a, 'static> = ()>"
pub use dyn_trait::Ty2;

// @has user/type.Ty3.html
// @has - '//*[@class="rust item-decl"]//code' "&'s (dyn ToString + 's)"
// FIXME(fmease): Hide default lifetime bound, render "&'s dyn ToString"
pub use dyn_trait::Ty3;

// @has user/fn.func0.html
// @has - '//pre[@class="rust item-decl"]' "func0(_: &dyn Fn())"
// FIXME(fmease): Show placeholder-lifetime bound, render "func0(_: &(dyn Fn() + '_))"
pub use dyn_trait::func0;

// @has user/fn.func1.html
// @has - '//pre[@class="rust item-decl"]' "func1<'func>(_: &(dyn Fn() + 'func))"
pub use dyn_trait::func1;
