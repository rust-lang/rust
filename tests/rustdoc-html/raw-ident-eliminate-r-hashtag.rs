#![crate_type="lib"]

pub mod internal {
    //@ has 'raw_ident_eliminate_r_hashtag/internal/type.mod.html'
    #[allow(non_camel_case_types)]
    pub struct r#mod;

    /// See [name], [other name]
    ///
    /// [name]: mod
    /// [other name]: crate::internal::mod
    //@ has 'raw_ident_eliminate_r_hashtag/internal/type.B.html' '//*a[@href="type.mod.html"]' 'name'
    //@ has 'raw_ident_eliminate_r_hashtag/internal/type.B.html' '//*a[@href="type.mod.html"]' 'other name'
    pub struct B;
}

/// See [name].
///
/// [name]: internal::mod
//@ has 'raw_ident_eliminate_r_hashtag/type.A.html' '//*a[@href="internal/type.mod.html"]' 'name'
pub struct A;
