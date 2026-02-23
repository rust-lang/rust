#![crate_name = "foo"]

//@ has 'foo/union.Union.html'
// Checking that there is a whitespace after `:`.
//@ has - '//*[@id="structfield.a"]/code' 'a: u8'
//@ has - '//*[@id="structfield.b"]/code' 'b: u32'
pub union Union {
    pub a: u8,
    /// tadam
    pub b: u32,
}
