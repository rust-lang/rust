#![feature(no_core)]
#![no_core]

pub enum Foo {
    // @set i8 = "$.index[*][?(@.docs=='i8' && @.kind=='field')].id"
    // @is       "$.index[*][?(@.docs=='i8' && @.kind=='field')].name" '"0"'
    // @is       "$.index[*][?(@.name=='V1')].inner.fields[*]" $i8
    // @is       "$.index[*][?(@.name=='V1')].inner.fields_stripped" false
    V1(
        /// i8
        i8,
    ),
    // @set u8 = "$.index[*][?(@.docs=='u8' && @.kind=='field')].id"
    // @is       "$.index[*][?(@.docs=='u8' && @.kind=='field')].name" '"1"'
    // @is       "$.index[*][?(@.name=='V2')].inner.fields[*]" $u8
    // @is       "$.index[*][?(@.name=='V2')].inner.fields_stripped" true
    V2(
        #[doc(hidden)] u8,
        /// u8
        u8,
    ),
}
