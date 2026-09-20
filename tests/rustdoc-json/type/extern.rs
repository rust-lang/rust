#![feature(extern_types)]

extern "C" {
    /// No inner information
    pub type Foo;
}

//@ is "$.index[?(@.docs[0].text=='No inner information')].name" '"Foo"'
//@ is "$.index[?(@.docs[0].text=='No inner information')].inner" \"extern_type\"
