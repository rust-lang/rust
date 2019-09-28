#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct ConstString<const T: &'static str>;
struct ConstBytes<const T: &'static [u8]>;

pub fn main() {
    let _: ConstString<"Hello"> = ConstString::<"World">; //~ ERROR mismatched types
    let _: ConstBytes<b"AAA"> = ConstBytes::<{&[0x41, 0x41, 0x41]}>;
    let _: ConstBytes<b"AAA"> = ConstBytes::<b"BBB">; //~ ERROR mismatched types
}
