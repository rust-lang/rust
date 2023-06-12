#![crate_name = "user"]

// In each test case, we include the trailing semicolon to ensure that nothing extra comes
// after the type like an unwanted outlives-bound.

// aux-crate:dyn_trait=dyn_trait.rs
// edition:2021

// @has user/type.Ty0.html
// @has - '//*[@class="rust item-decl"]//code' "dyn for<'any> FnOnce(&'any str) -> bool;"
pub use dyn_trait::Ty0;

// @has user/type.Ty1.html
// @has - '//*[@class="rust item-decl"]//code' "dyn Display + 'obj;"
pub use dyn_trait::Ty1;

// @has user/type.Ty2.html
// @has - '//*[@class="rust item-decl"]//code' "dyn for<'a, 'r> Container<'r, Item<'a, 'static> = ()>;"
pub use dyn_trait::Ty2;

// @has user/type.Ty3.html
// @has - '//*[@class="rust item-decl"]//code' "&'s dyn ToString;"
pub use dyn_trait::Ty3;

// Below we check if we correctly elide trait-object lifetime bounds if they coincide with their
// default (known as "object lifetime default" or "default trait object lifetime").

// @has user/fn.lbwel.html
// @has - '//pre[@class="rust item-decl"]' "lbwel(_: &dyn Fn())"
pub use dyn_trait::late_bound_wrapped_elided as lbwel;
// @has user/fn.lbwl0.html
// has - '//pre[@class="rust item-decl"]' "lbwl0<'f>(_: &mut (dyn Fn() + 'f))"
pub use dyn_trait::late_bound_wrapped_late0 as lbwl0;
// @has user/fn.lbwd0.html
// has - '//pre[@class="rust item-decl"]' "lbwd0<'f>(_: &'f mut dyn Fn())"
pub use dyn_trait::late_bound_wrapped_defaulted0 as lbwd0;
// @has user/type.EarlyBoundWrappedDefaulted0.html
// @has - '//*[@class="rust item-decl"]//code' "Ref<'x, dyn Trait>;"
pub use dyn_trait::EarlyBoundWrappedDefaulted0;
// @has user/type.EarlyBoundWrappedDefaulted1.html
// @has - '//*[@class="rust item-decl"]//code' "&'x dyn Trait;"
pub use dyn_trait::EarlyBoundWrappedDefaulted1;
// @has user/type.EarlyBoundWrappedEarly.html
// @has - '//*[@class="rust item-decl"]//code' "Ref<'x, dyn Trait + 'y>"
pub use dyn_trait::EarlyBoundWrappedEarly;
// @has user/type.EarlyBoundWrappedStatic.html
// @has - '//*[@class="rust item-decl"]//code' "Ref<'x, dyn Trait + 'static>"
pub use dyn_trait::EarlyBoundWrappedStatic;
// @has user/fn.lbwd1.html
// @has - '//pre[@class="rust item-decl"]' "lbwd1<'l>(_: Ref<'l, dyn Trait>)"
pub use dyn_trait::late_bound_wrapped_defaulted1 as lbwd1;
// @has user/fn.lbwl1.html
// @has - '//pre[@class="rust item-decl"]' "lbwl1<'l, 'm>(_: Ref<'l, dyn Trait + 'm>)"
pub use dyn_trait::late_bound_wrapped_late1 as lbwl1;
// @has user/fn.lbwe.html
// @has - '//pre[@class="rust item-decl"]' "lbwe<'e, 'l>(_: Ref<'l, dyn Trait + 'e>)"
pub use dyn_trait::late_bound_wrapped_early as lbwe;
// @has user/fn.ebwd.html
// @has - '//pre[@class="rust item-decl"]' "ebwd(_: Ref<'_, dyn Trait>)"
pub use dyn_trait::elided_bound_wrapped_defaulted as ebwd;
// @has user/type.StaticBoundWrappedDefaulted0.html
// @has - '//*[@class="rust item-decl"]//code' "Ref<'static, dyn Trait>;"
pub use dyn_trait::StaticBoundWrappedDefaulted0;
// @has user/type.StaticBoundWrappedDefaulted1.html
// @has - '//*[@class="rust item-decl"]//code' "&'static dyn Trait;"
pub use dyn_trait::StaticBoundWrappedDefaulted1;
// @has user/type.AmbiguousBoundWrappedEarly0.html
// @has - '//*[@class="rust item-decl"]//code' "AmbiguousBoundWrapper<'s, 'r, dyn Trait + 's>;"
pub use dyn_trait::AmbiguousBoundWrappedEarly0;
// @has user/type.AmbiguousBoundWrappedEarly1.html
// @has - '//*[@class="rust item-decl"]//code' "AmbiguousBoundWrapper<'s, 'r, dyn Trait + 'r>;"
pub use dyn_trait::AmbiguousBoundWrappedEarly1;
// @has user/type.AmbiguousBoundWrappedStatic.html
// @has - '//*[@class="rust item-decl"]//code' "AmbiguousBoundWrapper<'q, 'q, dyn Trait + 'static>;"
pub use dyn_trait::AmbiguousBoundWrappedStatic;

// @has user/type.NoBoundsWrappedDefaulted.html
// @has - '//*[@class="rust item-decl"]//code' "Box<dyn Trait, Global>;"
pub use dyn_trait::NoBoundsWrappedDefaulted;
// @has user/type.NoBoundsWrappedEarly.html
// @has - '//*[@class="rust item-decl"]//code' "Box<dyn Trait + 'e, Global>;"
pub use dyn_trait::NoBoundsWrappedEarly;
// @has user/fn.nbwl.html
// @has - '//pre[@class="rust item-decl"]' "nbwl<'l>(_: Box<dyn Trait + 'l, Global>)"
pub use dyn_trait::no_bounds_wrapped_late as nbwl;
// @has user/fn.nbwel.html
// @has - '//pre[@class="rust item-decl"]' "nbwel(_: Box<dyn Trait + '_, Global>)"
// NB: It might seem counterintuitive to display the explicitly elided lifetime `'_` here instead of
// eliding it but this behavior is correct: The default is `'static` here which != `'_`.
pub use dyn_trait::no_bounds_wrapped_elided as nbwel;

// @has user/type.BareNoBoundsDefaulted.html
// @has - '//*[@class="rust item-decl"]//code' "dyn Trait;"
pub use dyn_trait::BareNoBoundsDefaulted;
// @has user/type.BareNoBoundsEarly.html
// @has - '//*[@class="rust item-decl"]//code' "dyn Trait + 'p;"
pub use dyn_trait::BareNoBoundsEarly;
// @has user/type.BareEarlyBoundDefaulted0.html
// @has - '//*[@class="rust item-decl"]//code' "dyn EarlyBoundTrait0<'u>;"
pub use dyn_trait::BareEarlyBoundDefaulted0;
// @has user/type.BareEarlyBoundDefaulted1.html
// @has - '//*[@class="rust item-decl"]//code' "dyn for<'any> EarlyBoundTrait0<'any>;"
pub use dyn_trait::BareEarlyBoundDefaulted1;
// @has user/type.BareEarlyBoundDefaulted2.html
// @has - '//*[@class="rust item-decl"]//code' "dyn EarlyBoundTrait1<'static, 'w>;"
pub use dyn_trait::BareEarlyBoundDefaulted2;
// @has user/type.BareEarlyBoundEarly.html
// @has - '//*[@class="rust item-decl"]//code' "dyn EarlyBoundTrait0<'i> + 'j;"
pub use dyn_trait::BareEarlyBoundEarly;
// @has user/type.BareEarlyBoundStatic.html
// @has - '//*[@class="rust item-decl"]//code' "dyn EarlyBoundTrait0<'i> + 'static;"
pub use dyn_trait::BareEarlyBoundStatic;
// @has user/type.BareStaticBoundDefaulted.html
// @has - '//*[@class="rust item-decl"]//code' "dyn StaticBoundTrait;"
pub use dyn_trait::BareStaticBoundDefaulted;
// @has user/type.BareHigherRankedBoundDefaulted0.html
// @has - '//*[@class="rust item-decl"]//code' "dyn HigherRankedBoundTrait0;"
pub use dyn_trait::BareHigherRankedBoundDefaulted0;
// @has user/type.BareHigherRankedBoundDefaulted1.html
// @has - '//*[@class="rust item-decl"]//code' "dyn HigherRankedBoundTrait1<'r>;"
pub use dyn_trait::BareHigherRankedBoundDefaulted1;
// @has user/type.BareAmbiguousBoundEarly0.html
// @has - '//*[@class="rust item-decl"]//code' "dyn AmbiguousBoundTrait<'m, 'n> + 'm;"
pub use dyn_trait::BareAmbiguousBoundEarly0;
// @has user/type.BareAmbiguousBoundEarly1.html
// @has - '//*[@class="rust item-decl"]//code' "dyn AmbiguousBoundTrait<'m, 'n> + 'n;"
pub use dyn_trait::BareAmbiguousBoundEarly1;
// @has user/type.BareAmbiguousBoundStatic.html
// @has - '//*[@class="rust item-decl"]//code' "dyn AmbiguousBoundTrait<'o, 'o> + 'static;"
pub use dyn_trait::BareAmbiguousBoundStatic;
