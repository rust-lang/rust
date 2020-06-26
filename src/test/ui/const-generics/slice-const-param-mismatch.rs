#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

struct ConstString<const T: &'static str>;
struct ConstBytes<const T: &'static [u8]>;

pub fn main() {
    let _: ConstString<"Hello"> = ConstString::<"Hello">;
    let _: ConstString<"Hello"> = ConstString::<"World">; //~ ERROR mismatched types
    let _: ConstString<"ℇ㇈↦"> = ConstString::<"ℇ㇈↦">;
    let _: ConstString<"ℇ㇈↦"> = ConstString::<"ℇ㇈↥">; //~ ERROR mismatched types
    let _: ConstBytes<b"AAA"> = ConstBytes::<{&[0x41, 0x41, 0x41]}>;
    let _: ConstBytes<b"AAA"> = ConstBytes::<b"BBB">; //~ ERROR mismatched types
}
