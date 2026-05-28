//@ revisions: min adt_const_params full

#![cfg_attr(full, feature(adt_const_params, unsized_const_params))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(adt_const_params, feature(adt_const_params))]
#![cfg_attr(adt_const_params, allow(incomplete_features))]

struct ConstString<const T: &'static str>;
//[min]~^ ERROR
//[adt_const_params]~^^ ERROR
struct ConstBytes<const T: &'static [u8]>;
//[min]~^ ERROR
//[adt_const_params]~^^ ERROR

pub fn main() {
    let _: ConstString<"Hello"> = ConstString::<"Hello">;
    let _: ConstString<"Hello"> = ConstString::<"World">; //~ ERROR mismatched types
    let _: ConstString<"ℇ㇈↦"> = ConstString::<"ℇ㇈↦">;
    let _: ConstString<"ℇ㇈↦"> = ConstString::<"ℇ㇈↥">; //~ ERROR mismatched types
    let _: ConstBytes<b"AAA"> = ConstBytes::<{ &[0x41, 0x41, 0x41] }>;
    let _: ConstBytes<b"AAA"> = ConstBytes::<b"BBB">; //~ ERROR mismatched types
}
