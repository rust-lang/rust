bitflags! {
    #[derive(HashStable, TyEncodable, TyDecodable)]
    pub struct VariantFlags: u8 {
        const NO_VARIANT_FLAGS = 0;
        /// Indicates whether the field list of this variant is `#[non_exhaustive]`.
        const IS_FIELD_LIST_NON_EXHAUSTIVE = 1 << 0;
        /// Indicates whether this variant was obtained as part of recovering from
        /// a syntactic error. May be incomplete or bogus.
        const IS_RECOVERED = 1 << 1;
    }
}
