#![feature(staged_api, rustc_attrs, associated_type_defaults)]

#[stable(feature = "default_body_trait_feature", since = "1.0.0")]
pub trait TraitWithDefaults {
    //@ is "$.index[?(@.docs=='method with unstable default body')].inner.function.has_body" true
    //@ is "$.index[?(@.docs=='method with unstable default body')].inner.function.default_unstable.feature" '"method_default_body_feature"'
    //@ is "$.index[?(@.docs=='method with unstable default body')].attrs" []
    /// method with unstable default body
    #[stable(feature = "default_body_method_feature", since = "1.1.0")]
    #[rustc_default_body_unstable(feature = "method_default_body_feature", issue = "none")]
    fn method_with_unstable_default() {}

    //@ is "$.index[?(@.docs=='required method without default body')].inner.function.has_body" false
    //@ is "$.index[?(@.docs=='required method without default body')].inner.function.default_unstable" null
    /// required method without default body
    #[stable(feature = "required_method_feature", since = "1.2.0")]
    fn required_method();

    //@ is "$.index[?(@.docs=='method with stable default body')].inner.function.has_body" true
    //@ is "$.index[?(@.docs=='method with stable default body')].inner.function.default_unstable" null
    /// method with stable default body
    #[stable(feature = "stable_default_method_feature", since = "1.3.0")]
    fn method_with_stable_default() {}

    //@ is "$.index[?(@.docs=='associated constant with unstable default value')].inner.assoc_const.value" '"0"'
    //@ is "$.index[?(@.docs=='associated constant with unstable default value')].inner.assoc_const.default_unstable.feature" '"assoc_const_default_value_feature"'
    //@ is "$.index[?(@.docs=='associated constant with unstable default value')].attrs" []
    /// associated constant with unstable default value
    #[stable(feature = "assoc_const_with_unstable_default_feature", since = "1.4.0")]
    #[rustc_default_body_unstable(feature = "assoc_const_default_value_feature", issue = "none")]
    const UNSTABLE_DEFAULT_CONST: usize = 0;

    //@ is "$.index[?(@.docs=='required associated constant')].inner.assoc_const.value" null
    //@ is "$.index[?(@.docs=='required associated constant')].inner.assoc_const.default_unstable" null
    /// required associated constant
    #[stable(feature = "required_assoc_const_feature", since = "1.5.0")]
    const REQUIRED_CONST: usize;

    //@ is "$.index[?(@.docs=='associated type with unstable default type')].inner.assoc_type.default_unstable.feature" '"assoc_type_default_type_feature"'
    //@ is "$.index[?(@.docs=='associated type with unstable default type')].attrs" []
    /// associated type with unstable default type
    #[stable(feature = "assoc_type_with_unstable_default_feature", since = "1.6.0")]
    #[rustc_default_body_unstable(feature = "assoc_type_default_type_feature", issue = "none")]
    type UnstableDefaultType = usize;

    //@ is "$.index[?(@.docs=='required associated type')].inner.assoc_type.type" null
    //@ is "$.index[?(@.docs=='required associated type')].inner.assoc_type.default_unstable" null
    /// required associated type
    #[stable(feature = "required_assoc_type_feature", since = "1.7.0")]
    type RequiredType;
}

#[stable(feature = "default_body_impl_target_feature", since = "2.0.0")]
pub struct ImplTarget;

// Impl items provide their own definitions, so they do not use the trait's unstable defaults.
#[stable(feature = "default_body_impl_feature", since = "2.1.0")]
impl TraitWithDefaults for ImplTarget {
    //@ is "$.index[?(@.docs=='impl override for unstable default body')].inner.function.default_unstable" null
    /// impl override for unstable default body
    fn method_with_unstable_default() {}

    fn required_method() {}

    //@ is "$.index[?(@.docs=='impl override for unstable default value')].inner.assoc_const.default_unstable" null
    /// impl override for unstable default value
    const UNSTABLE_DEFAULT_CONST: usize = 1;

    const REQUIRED_CONST: usize = 2;

    //@ is "$.index[?(@.docs=='impl override for unstable default type')].inner.assoc_type.default_unstable" null
    /// impl override for unstable default type
    type UnstableDefaultType = u8;

    type RequiredType = ();
}
