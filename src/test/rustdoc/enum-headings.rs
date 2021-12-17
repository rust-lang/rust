#![crate_name = "foo"]
// @has foo/enum.Token.html
/// A token!
/// # First
/// Some following text...
// @has - '//h2[@id="enum.Token.first"]' "First"
pub enum Token {
    /// A declaration!
    /// # Variant-First
    /// Some following text...
    // @has - '//h4[@id="variant.Declaration.variant-first"]' "Variant-First"
    Declaration {
        /// A version!
        /// # Variant-Field-First
        /// Some following text...
        // @has - '//h5[@id="structfield.version.variant-field-first"]' "Variant-Field-First"
        version: String,
    },
    /// A Zoople!
    /// # Variant-First
    Zoople(
        // @has - '//h5[@id="structfield.0.variant-tuple-field-first"]' "Variant-Tuple-Field-First"
        /// Zoople's first variant!
        /// # Variant-Tuple-Field-First
        /// Some following text...
        usize,
    ),
    /// Unfinished business!
    /// # Non-Exhaustive-First
    /// Some following text...
    // @has - '//h4[@id="variant.Unfinished.non-exhaustive-first"]' "Non-Exhaustive-First"
    #[non_exhaustive]
    Unfinished {
        /// This is x.
        /// # X-First
        /// Some following text...
        // @has - '//h5[@id="structfield.x.x-first"]' "X-First"
        x: usize,
    },
}
