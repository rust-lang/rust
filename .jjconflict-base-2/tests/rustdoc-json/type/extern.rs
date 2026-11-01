#![feature(extern_types)]

extern "C" {
    /// No inner information
    pub type Foo;
}

//@ is "$.index[?(@.docs=='No inner information')].name" '"Foo"'
//@ is "$.index[?(@.docs=='No inner information')].inner" \"extern_type\"
