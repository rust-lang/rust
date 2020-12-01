// ignore-tidy-linelength

#![crate_type="lib"]

pub mod internal {
    // @has 'raw_ident_eliminate_r_hashtag/internal/struct.mod.html'
    pub struct r#mod;

    /// See [name], [other name]
    ///
    /// [name]: mod
    /// [other name]: crate::internal::mod
    // @has 'raw_ident_eliminate_r_hashtag/internal/struct.B.html' '//*a[@href="../../raw_ident_eliminate_r_hashtag/internal/struct.mod.html"]' 'name'
    // @has 'raw_ident_eliminate_r_hashtag/internal/struct.B.html' '//*a[@href="../../raw_ident_eliminate_r_hashtag/internal/struct.mod.html"]' 'other name'
    pub struct B;
}

/// See [name].
///
/// [name]: internal::mod
// @has 'raw_ident_eliminate_r_hashtag/struct.A.html' '//*a[@href="../raw_ident_eliminate_r_hashtag/internal/struct.mod.html"]' 'name'
pub struct A;
