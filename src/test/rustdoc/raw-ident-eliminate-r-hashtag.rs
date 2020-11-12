// ignore-tidy-linelength

pub mod internal {
    pub struct r#mod;

    /// See [name], [other name]
    ///
    /// [name]: mod
    /// [other name]: ../internal/struct.mod.html
    // @has 'raw_ident_eliminate_r_hashtag/internal/struct.B.html' '//a[@href="../raw_ident_eliminate_r_hashtag/internal/struct.mod.html"]' 'name'
    // @has 'raw_ident_eliminate_r_hashtag/internal/struct.B.html' '//a[@href="../raw_ident_eliminate_r_hashtag/internal/struct.mod.html"]' 'other name'
    pub struct B;
}

/// See [name].
///
/// [name]: internal::mod
// @has 'raw_ident_eliminate_r_hashtag/struct.A.html' '//a[@href="../raw_ident_eliminate_r_hashtag/internal/struct.mod.html"]' 'name'
struct A;
