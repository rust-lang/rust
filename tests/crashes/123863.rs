//@ known-bug: #123863
const fn concat_strs<const A: &'static str>() -> &'static str {
    struct Inner<const A: &'static str>;
    Inner::concat_strs::<"a">::A
}
pub fn main() {}
