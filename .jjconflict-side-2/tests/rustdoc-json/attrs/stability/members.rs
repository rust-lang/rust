#![feature(staged_api)]

// Mirrors standard-library cases where stable parent items have unstable variants or fields.

#[stable(feature = "stable_enum_feature", since = "1.0.0")]
pub enum StableEnumWithUnstableVariant {
    StableVariant,

    //@ is "$.index[?(@.name=='UnstableVariant')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='UnstableVariant')].stability.feature" '"unstable_variant_feature"'
    //@ !has "$.index[?(@.name=='UnstableVariant')].stability.since"
    #[unstable(feature = "unstable_variant_feature", issue = "none")]
    UnstableVariant,
}

#[stable(feature = "stable_struct_feature", since = "2.0.0")]
pub struct StableStructWithUnstableField {
    pub stable_field: usize,

    //@ is "$.index[?(@.name=='unstable_field')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='unstable_field')].stability.feature" '"unstable_field_feature"'
    //@ !has "$.index[?(@.name=='unstable_field')].stability.since"
    #[unstable(feature = "unstable_field_feature", issue = "none")]
    pub unstable_field: usize,
}

#[stable(feature = "stable_union_with_unstable_field", since = "2.5.0")]
pub union StableUnionWithUnstableField {
    pub stable_union_field: usize,

    //@ is "$.index[?(@.name=='unstable_union_field')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='unstable_union_field')].stability.feature" '"unstable_union_field_feature"'
    //@ !has "$.index[?(@.name=='unstable_union_field')].stability.since"
    #[unstable(feature = "unstable_union_field_feature", issue = "none")]
    pub unstable_union_field: usize,
}

#[stable(feature = "stable_tuple_struct_feature", since = "3.0.0")]
pub struct StableTupleStructWithUnstableField(
    pub usize,
    //@ is "$.index[?(@.docs=='unstable tuple struct field')].stability.level" '"unstable"'
    //@ is "$.index[?(@.docs=='unstable tuple struct field')].stability.feature" '"unstable_tuple_struct_field_feature"'
    //@ !has "$.index[?(@.docs=='unstable tuple struct field')].stability.since"
    /// unstable tuple struct field
    #[unstable(feature = "unstable_tuple_struct_field_feature", issue = "none")]
    pub usize,
);

#[stable(feature = "stable_enum_field_variants_feature", since = "4.0.0")]
pub enum StableEnumWithFieldVariants {
    #[stable(feature = "stable_tuple_variant_feature", since = "4.1.0")]
    TupleVariant(
        usize,
        //@ is "$.index[?(@.docs=='unstable tuple variant field')].stability.level" '"unstable"'
        //@ is "$.index[?(@.docs=='unstable tuple variant field')].stability.feature" '"unstable_tuple_variant_field_feature"'
        //@ !has "$.index[?(@.docs=='unstable tuple variant field')].stability.since"
        /// unstable tuple variant field
        #[unstable(feature = "unstable_tuple_variant_field_feature", issue = "none")]
        usize,
    ),

    #[stable(feature = "stable_struct_variant_feature", since = "4.3.0")]
    StructVariant {
        stable_struct_variant_field: usize,

        //@ is "$.index[?(@.name=='unstable_struct_variant_field')].stability.level" '"unstable"'
        //@ is "$.index[?(@.name=='unstable_struct_variant_field')].stability.feature" '"unstable_struct_variant_field_feature"'
        //@ !has "$.index[?(@.name=='unstable_struct_variant_field')].stability.since"
        #[unstable(feature = "unstable_struct_variant_field_feature", issue = "none")]
        unstable_struct_variant_field: usize,
    },
}
