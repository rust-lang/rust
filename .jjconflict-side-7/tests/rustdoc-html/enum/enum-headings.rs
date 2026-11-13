#![crate_name = "foo"]
//@ has foo/enum.Token.html
/// A token!
/// # First
/// Some following text...
//@ has - '//h2[@id="first"]' "First"
pub enum Token {
    /// A declaration!
    /// # Variant-First
    /// Some following text...
    //@ has - '//h4[@id="variant-first"]' "Variant-First"
    Declaration {
        /// A version!
        /// # Variant-Field-First
        /// Some following text...
        //@ has - '//h5[@id="variant-field-first"]' "Variant-Field-First"
        version: String,
    },
    /// A Zoople!
    /// # Variant-First
    Zoople(
        //@ has - '//h5[@id="variant-tuple-field-first"]' "Variant-Tuple-Field-First"
        /// Zoople's first variant!
        /// # Variant-Tuple-Field-First
        /// Some following text...
        usize,
    ),
    /// Unfinished business!
    /// # Non-Exhaustive-First
    /// Some following text...
    //@ has - '//h4[@id="non-exhaustive-first"]' "Non-Exhaustive-First"
    #[non_exhaustive]
    Unfinished {
        /// This is x.
        /// # X-First
        /// Some following text...
        //@ has - '//h5[@id="x-first"]' "X-First"
        x: usize,
    },
}
