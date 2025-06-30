//@ arg _1_1_0 .index[] | select(.docs == "1.1.0")
//@ arg _2_1_0 .index[] | select(.docs == "2.1.0")
//@ arg _2_1_1 .index[] | select(.docs == "2.1.1")
//@ arg _2_2_1 .index[] | select(.docs == "2.2.1")
//@ arg _2_3_0 .index[] | select(.docs == "2.3.0")
//@ arg _3_1_1 .index[] | select(.docs == "3.1.1")
//@ arg _3_1_2 .index[] | select(.docs == "3.1.2")
//@ arg _3_2_0 .index[] | select(.docs == "3.2.0")
//@ arg _3_2_2 .index[] | select(.docs == "3.2.2")
//@ arg _3_3_0 .index[] | select(.docs == "3.3.0")
//@ arg _3_3_1 .index[] | select(.docs == "3.3.1")

pub enum EnumWithStrippedTupleVariants {
    //@ jq .index[] | select(.name == "None").inner.variant.kind?.tuple | length == 0
    None(),

    //@ jq .index[] | select(.name == "One").inner.variant.kind?.tuple == [$_1_1_0.id]
    One(/** 1.1.0*/ bool),
    //@ jq .index[] | select(.name == "OneHidden").inner.variant.kind?.tuple == [null]
    OneHidden(#[doc(hidden)] bool),

    //@ jq .index[] | select(.name == "Two").inner.variant.kind?.tuple == [$_2_1_0.id, $_2_1_1.id]
    Two(/** 2.1.0*/ bool, /** 2.1.1*/ bool),
    //@ jq .index[] | select(.name == "TwoLeftHidden").inner.variant.kind?.tuple == [null, $_2_2_1.id]
    TwoLeftHidden(#[doc(hidden)] bool, /** 2.2.1*/ bool),
    //@ jq .index[] | select(.name == "TwoRightHidden").inner.variant.kind?.tuple == [$_2_3_0.id, null]
    TwoRightHidden(/** 2.3.0*/ bool, #[doc(hidden)] bool),
    //@ jq .index[] | select(.name == "TwoBothHidden").inner.variant.kind?.tuple == [null, null]
    TwoBothHidden(#[doc(hidden)] bool, #[doc(hidden)] bool),

    //@ jq .index[] | select(.name == "Three1").inner.variant.kind?.tuple == [null, $_3_1_1.id, $_3_1_2.id]
    Three1(#[doc(hidden)] bool, /** 3.1.1*/ bool, /** 3.1.2*/ bool),
    //@ jq .index[] | select(.name == "Three2").inner.variant.kind?.tuple == [$_3_2_0.id, null, $_3_2_2.id]
    Three2(/** 3.2.0*/ bool, #[doc(hidden)] bool, /** 3.2.2*/ bool),
    //@ jq .index[] | select(.name == "Three3").inner.variant.kind?.tuple == [$_3_3_0.id, $_3_3_1.id, null]
    Three3(/** 3.3.0*/ bool, /** 3.3.1*/ bool, #[doc(hidden)] bool),
}

//@ jq $_1_1_0.name == "0"
//@ jq $_2_1_0.name == "0"
//@ jq $_2_1_1.name == "1"
//@ jq $_2_2_1.name == "1"
//@ jq $_2_3_0.name == "0"
//@ jq $_3_1_1.name == "1"
//@ jq $_3_1_2.name == "2"
//@ jq $_3_2_0.name == "0"
//@ jq $_3_2_2.name == "2"
//@ jq $_3_3_0.name == "0"
//@ jq $_3_3_1.name == "1"

//@ jq [[$_1_1_0, $_2_1_0, $_2_1_1, $_2_2_1, $_2_3_0, $_3_1_1, $_3_1_2, $_3_2_0, $_3_2_2, $_3_3_0, $_3_3_1][].inner.struct_field.primitive? == "bool"] | all
