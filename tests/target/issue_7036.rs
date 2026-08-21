// Doc comments inside macro calls should not prevent formatting.

foo!(
    /// Doc
    A,
    B,
);

foo!(
    #[doc = ""]
    /// Doc
    A,
    B,
);

foo!(
    /// Doc
    #[doc = ""]
    A,
    B,
);
