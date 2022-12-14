#![crate_name = "foo"]
// @has foo/enum.E1.html '//a/@href' 'enum.E1.html#variant.A'

/// [Self::A::b]
pub enum E1 {
    A { b: usize }
}

// @has foo/enum.E2.html '//a/@href' 'enum.E2.html#variant.A'

/// [Self::A::b]
pub enum E2 {
    A { b: usize }
}
