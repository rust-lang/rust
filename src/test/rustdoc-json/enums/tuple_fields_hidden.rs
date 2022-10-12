#![feature(no_core)]
#![no_core]

// @set 1.1.0 = "$.index[*][?(@.docs=='1.1.0')].id"
// @set 2.1.0 = "$.index[*][?(@.docs=='2.1.0')].id"
// @set 2.1.1 = "$.index[*][?(@.docs=='2.1.1')].id"
// @set 2.2.1 = "$.index[*][?(@.docs=='2.2.1')].id"
// @set 2.3.0 = "$.index[*][?(@.docs=='2.3.0')].id"
// @set 3.1.1 = "$.index[*][?(@.docs=='3.1.1')].id"
// @set 3.1.2 = "$.index[*][?(@.docs=='3.1.2')].id"
// @set 3.2.0 = "$.index[*][?(@.docs=='3.2.0')].id"
// @set 3.2.2 = "$.index[*][?(@.docs=='3.2.2')].id"
// @set 3.3.0 = "$.index[*][?(@.docs=='3.3.0')].id"
// @set 3.3.1 = "$.index[*][?(@.docs=='3.3.1')].id"

pub enum EnumWithStrippedTupleVariants {
    // @is    "$.index[*][?(@.name=='None')].inner.variant_kind" '"tuple"'
    // @count "$.index[*][?(@.name=='None')].inner.variant_inner[*]" 0
    None(),

    // @is    "$.index[*][?(@.name=='One')].inner.variant_kind" '"tuple"'
    // @count "$.index[*][?(@.name=='One')].inner.variant_inner[*]" 1
    // @is    "$.index[*][?(@.name=='One')].inner.variant_inner[0]" $1.1.0
    One(/** 1.1.0*/ bool),
    // @is    "$.index[*][?(@.name=='OneHidden')].inner.variant_kind" '"tuple"'
    // @count "$.index[*][?(@.name=='OneHidden')].inner.variant_inner[*]" 1
    // @is    "$.index[*][?(@.name=='OneHidden')].inner.variant_inner[0]" null
    OneHidden(#[doc(hidden)] bool),

    // @is    "$.index[*][?(@.name=='Two')].inner.variant_kind" '"tuple"'
    // @count "$.index[*][?(@.name=='Two')].inner.variant_inner[*]" 2
    // @is    "$.index[*][?(@.name=='Two')].inner.variant_inner[0]" $2.1.0
    // @is    "$.index[*][?(@.name=='Two')].inner.variant_inner[1]" $2.1.1
    Two(/** 2.1.0*/ bool, /** 2.1.1*/ bool),
    // @is    "$.index[*][?(@.name=='TwoLeftHidden')].inner.variant_kind" '"tuple"'
    // @count "$.index[*][?(@.name=='TwoLeftHidden')].inner.variant_inner[*]" 2
    // @is    "$.index[*][?(@.name=='TwoLeftHidden')].inner.variant_inner[0]" null
    // @is    "$.index[*][?(@.name=='TwoLeftHidden')].inner.variant_inner[1]" $2.2.1
    TwoLeftHidden(#[doc(hidden)] bool, /** 2.2.1*/ bool),
    // @is    "$.index[*][?(@.name=='TwoRightHidden')].inner.variant_kind" '"tuple"'
    // @count "$.index[*][?(@.name=='TwoRightHidden')].inner.variant_inner[*]" 2
    // @is    "$.index[*][?(@.name=='TwoRightHidden')].inner.variant_inner[0]" $2.3.0
    // @is    "$.index[*][?(@.name=='TwoRightHidden')].inner.variant_inner[1]" null
    TwoRightHidden(/** 2.3.0*/ bool, #[doc(hidden)] bool),
    // @is    "$.index[*][?(@.name=='TwoBothHidden')].inner.variant_kind" '"tuple"'
    // @count "$.index[*][?(@.name=='TwoBothHidden')].inner.variant_inner[*]" 2
    // @is    "$.index[*][?(@.name=='TwoBothHidden')].inner.variant_inner[0]" null
    // @is    "$.index[*][?(@.name=='TwoBothHidden')].inner.variant_inner[1]" null
    TwoBothHidden(#[doc(hidden)] bool, #[doc(hidden)] bool),

    // @is    "$.index[*][?(@.name=='Three1')].inner.variant_kind" '"tuple"'
    // @count "$.index[*][?(@.name=='Three1')].inner.variant_inner[*]" 3
    // @is    "$.index[*][?(@.name=='Three1')].inner.variant_inner[0]" null
    // @is    "$.index[*][?(@.name=='Three1')].inner.variant_inner[1]" $3.1.1
    // @is    "$.index[*][?(@.name=='Three1')].inner.variant_inner[2]" $3.1.2
    Three1(#[doc(hidden)] bool, /** 3.1.1*/ bool, /** 3.1.2*/ bool),
    // @is    "$.index[*][?(@.name=='Three2')].inner.variant_kind" '"tuple"'
    // @count "$.index[*][?(@.name=='Three2')].inner.variant_inner[*]" 3
    // @is    "$.index[*][?(@.name=='Three2')].inner.variant_inner[0]" $3.2.0
    // @is    "$.index[*][?(@.name=='Three2')].inner.variant_inner[1]" null
    // @is    "$.index[*][?(@.name=='Three2')].inner.variant_inner[2]" $3.2.2
    Three2(/** 3.2.0*/ bool, #[doc(hidden)] bool, /** 3.2.2*/ bool),
    // @is    "$.index[*][?(@.name=='Three3')].inner.variant_kind" '"tuple"'
    // @count "$.index[*][?(@.name=='Three3')].inner.variant_inner[*]" 3
    // @is    "$.index[*][?(@.name=='Three3')].inner.variant_inner[0]" $3.3.0
    // @is    "$.index[*][?(@.name=='Three3')].inner.variant_inner[1]" $3.3.1
    // @is    "$.index[*][?(@.name=='Three3')].inner.variant_inner[2]" null
    Three3(/** 3.3.0*/ bool, /** 3.3.1*/ bool, #[doc(hidden)] bool),
}


// @is "$.index[*][?(@.docs=='1.1.0')].name" '"0"'
// @is "$.index[*][?(@.docs=='2.1.0')].name" '"0"'
// @is "$.index[*][?(@.docs=='2.1.1')].name" '"1"'
// @is "$.index[*][?(@.docs=='2.2.1')].name" '"1"'
// @is "$.index[*][?(@.docs=='2.3.0')].name" '"0"'
// @is "$.index[*][?(@.docs=='3.1.1')].name" '"1"'
// @is "$.index[*][?(@.docs=='3.1.2')].name" '"2"'
// @is "$.index[*][?(@.docs=='3.2.0')].name" '"0"'
// @is "$.index[*][?(@.docs=='3.2.2')].name" '"2"'
// @is "$.index[*][?(@.docs=='3.3.0')].name" '"0"'
// @is "$.index[*][?(@.docs=='3.3.1')].name" '"1"'

// @is "$.index[*][?(@.docs=='1.1.0')].inner" '{"kind": "primitive", "inner": "bool"}'
// @is "$.index[*][?(@.docs=='2.1.0')].inner" '{"kind": "primitive", "inner": "bool"}'
// @is "$.index[*][?(@.docs=='2.1.1')].inner" '{"kind": "primitive", "inner": "bool"}'
// @is "$.index[*][?(@.docs=='2.2.1')].inner" '{"kind": "primitive", "inner": "bool"}'
// @is "$.index[*][?(@.docs=='2.3.0')].inner" '{"kind": "primitive", "inner": "bool"}'
// @is "$.index[*][?(@.docs=='3.1.1')].inner" '{"kind": "primitive", "inner": "bool"}'
// @is "$.index[*][?(@.docs=='3.1.2')].inner" '{"kind": "primitive", "inner": "bool"}'
// @is "$.index[*][?(@.docs=='3.2.0')].inner" '{"kind": "primitive", "inner": "bool"}'
// @is "$.index[*][?(@.docs=='3.2.2')].inner" '{"kind": "primitive", "inner": "bool"}'
// @is "$.index[*][?(@.docs=='3.3.0')].inner" '{"kind": "primitive", "inner": "bool"}'
// @is "$.index[*][?(@.docs=='3.3.1')].inner" '{"kind": "primitive", "inner": "bool"}'
