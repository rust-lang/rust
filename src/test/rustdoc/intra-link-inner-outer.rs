#![crate_name = "foo"]

pub enum A {}

/// Links to [outer A][A] and [outer B][B]
// @has foo/M/index.html '//*[@href="../foo/enum.A.html"]' 'outer A'
// @!has foo/M/index.html '//*[@href="../foo/struct.B.html"]' 'outer B'
// doesn't resolve unknown links
pub mod M {
    //! Links to [inner A][A] and [inner B][B]
    // @!has foo/M/index.html '//*[@href="../foo/enum.A.html"]' 'inner A'
    // @has foo/M/index.html '//*[@href="../foo/struct.B.html"]' 'inner B'
    pub struct B;
}

// distinguishes between links to inner and outer attributes
/// Links to [outer A][A]
// @has foo/N/index.html '//*[@href="../foo/enum.A.html"]' 'outer A'
pub mod N {
    //! Links to [inner A][A]
    // @has foo/N/index.html '//*[@href="../foo/struct.A.html"]' 'inner A'

    pub struct A;
}
