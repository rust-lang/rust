// @has intra_links/index.html
// @has - '//a/@href' '../intra_links/struct.ThisType.html'
// @has - '//a/@href' '../intra_links/struct.ThisType.html#method.this_method'
// @has - '//a/@href' '../intra_links/enum.ThisEnum.html'
// @has - '//a/@href' '../intra_links/enum.ThisEnum.html#ThisVariant.v'
// @has - '//a/@href' '../intra_links/trait.ThisTrait.html'
// @has - '//a/@href' '../intra_links/trait.ThisTrait.html#tymethod.this_associated_method'
// @has - '//a/@href' '../intra_links/trait.ThisTrait.html#associatedtype.ThisAssociatedType'
// @has - '//a/@href' '../intra_links/trait.ThisTrait.html#associatedconstant.THIS_ASSOCIATED_CONST'
// @has - '//a/@href' '../intra_links/trait.ThisTrait.html'
// @has - '//a/@href' '../intra_links/type.ThisAlias.html'
// @has - '//a/@href' '../intra_links/union.ThisUnion.html'
// @has - '//a/@href' '../intra_links/fn.this_function.html'
// @has - '//a/@href' '../intra_links/constant.THIS_CONST.html'
// @has - '//a/@href' '../intra_links/static.THIS_STATIC.html'
// @has - '//a/@href' '../intra_links/macro.this_macro.html'
// @has - '//a/@href' '../intra_links/trait.SoAmbiguous.html'
// @has - '//a/@href' '../intra_links/fn.SoAmbiguous.html'
//! In this crate we would like to link to:
//!
//! * [`ThisType`](ThisType)
//! * [`ThisType::this_method`](ThisType::this_method)
//! * [`ThisEnum`](ThisEnum)
//! * [`ThisEnum::ThisVariant`](ThisEnum::ThisVariant)
//! * [`ThisEnum::ThisVariantCtor`](ThisEnum::ThisVariantCtor)
//! * [`ThisTrait`](ThisTrait)
//! * [`ThisTrait::this_associated_method`](ThisTrait::this_associated_method)
//! * [`ThisTrait::ThisAssociatedType`](ThisTrait::ThisAssociatedType)
//! * [`ThisTrait::THIS_ASSOCIATED_CONST`](ThisTrait::THIS_ASSOCIATED_CONST)
//! * [`ThisAlias`](ThisAlias)
//! * [`ThisUnion`](ThisUnion)
//! * [`this_function`](this_function())
//! * [`THIS_CONST`](const@THIS_CONST)
//! * [`THIS_STATIC`](static@THIS_STATIC)
//! * [`this_macro`](this_macro!)
//!
//! In addition, there's some specifics we want to look at. There's [a trait called
//! SoAmbiguous][ambig-trait], but there's also [a function called SoAmbiguous][ambig-fn] too!
//! Whatever shall we do?
//!
//! [ambig-trait]: trait@SoAmbiguous
//! [ambig-fn]: SoAmbiguous()

#[macro_export]
macro_rules! this_macro {
    () => {};
}

pub struct ThisType;

impl ThisType {
    pub fn this_method() {}
}
pub enum ThisEnum { ThisVariant, ThisVariantCtor(u32), }
pub trait ThisTrait {
    type ThisAssociatedType;
    const THIS_ASSOCIATED_CONST: u8;
    fn this_associated_method();
}
pub type ThisAlias = Result<(), ()>;
pub union ThisUnion { this_field: usize, }

pub fn this_function() {}
pub const THIS_CONST: usize = 5usize;
pub static THIS_STATIC: usize = 5usize;

pub trait SoAmbiguous {}

#[allow(nonstandard_style)]
pub fn SoAmbiguous() {}


// @has - '//a/@href' '../intra_links/struct.ThisType.html'
// @has - '//a/@href' '../intra_links/struct.ThisType.html#method.this_method'
// @has - '//a/@href' '../intra_links/enum.ThisEnum.html'
// @has - '//a/@href' '../intra_links/enum.ThisEnum.html#ThisVariant.v'
/// Shortcut links for:
/// * [`ThisType`]
/// * [`ThisType::this_method`]
/// * [ThisEnum]
/// * [ThisEnum::ThisVariant]
pub struct SomeOtherType;
