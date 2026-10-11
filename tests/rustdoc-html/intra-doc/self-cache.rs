#![crate_name = "foo"]
//@ has foo/type.E1.html '//a/@href' 'type.E1.html#variant.A'

/// [Self::A::b]
pub enum E1 {
    A { b: usize }
}

//@ has foo/type.E2.html '//a/@href' 'type.E2.html#variant.A'

/// [Self::A::b]
pub enum E2 {
    A { b: usize }
}
