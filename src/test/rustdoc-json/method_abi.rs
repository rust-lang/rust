// @has method_abi.json "$.index[*][?(@.name=='Foo')]"
pub struct Foo;

impl Foo {
    // @has - "$.index[*][?(@.name=='abi_rust')].inner.abi" '"\"Rust\""'
    pub fn abi_rust() {}

    // @has - "$.index[*][?(@.name=='abi_c')].inner.abi" '"\"C\""'
    pub extern "C" fn abi_c() {}

    // @has - "$.index[*][?(@.name=='abi_system')].inner.abi" '"\"system\""'
    pub extern "system" fn abi_system() {}
}

// @has method_abi.json "$.index[*][?(@.name=='Bar')]"
pub trait Bar {
    // @has - "$.index[*][?(@.name=='trait_abi_rust')].inner.abi" '"\"Rust\""'
    fn trait_abi_rust();

    // @has - "$.index[*][?(@.name=='trait_abi_c')].inner.abi" '"\"C\""'
    extern "C" fn trait_abi_c();

    // @has - "$.index[*][?(@.name=='trait_abi_system')].inner.abi" '"\"system\""'
    extern "system" fn trait_abi_system();
}
