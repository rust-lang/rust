pub fn main() {
    let s: &str = "hello";
    let c: u8 = s[4]; //~ ERROR the type `str` cannot be indexed by `{integer}`
}
