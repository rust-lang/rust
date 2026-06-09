pub fn main() {
    let s: &str = "hello";
    let _: u8 = s[4]; //~ ERROR the type `str` cannot be indexed by `{integer}`
    let _ = s.get(4); //~ ERROR the type `str` cannot be indexed by `{integer}`
    let _ = s.get_unchecked(4); //~ ERROR the type `str` cannot be indexed by `{integer}`
    let _: u8 = s['c']; //~ ERROR the type `str` cannot be indexed by `char`
}
