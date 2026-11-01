// Doc comments inside macro calls should not prevent formatting.
//
// Each comment form below is tested three ways: on its own, after an
// attribute, and before an attribute.

foo!(
    // Only a comment
    A,
      B,
);

foo!(
    #[doc = ""]
    // Only a comment
    A,
      B,
);

foo!(
    // Only a comment
    #[doc = ""]
    A,
      B,
);


foo!(
    /// Outer line doc (exactly 3 slashes)
    A,
      B,
);

foo!(
    #[doc = ""]
    /// Outer line doc (exactly 3 slashes)
    A,
      B,
);

foo!(
    /// Outer line doc (exactly 3 slashes)
    #[doc = ""]
    A,
      B,
);


foo!(
    //// Only a comment
    A,
      B,
);

foo!(
    #[doc = ""]
    //// Only a comment
    A,
      B,
);

foo!(
    //// Only a comment
    #[doc = ""]
    A,
      B,
);


foo!(
    /* Only a comment */
    A,
      B,
);

foo!(
    #[doc = ""]
    /* Only a comment */
    A,
      B,
);

foo!(
    /* Only a comment */
    #[doc = ""]
    A,
      B,
);


foo!(
    /** Outer block doc (exactly 2 asterisks) */
    A,
      B,
);

foo!(
    #[doc = ""]
    /** Outer block doc (exactly 2 asterisks) */
    A,
      B,
);

foo!(
    /** Outer block doc (exactly 2 asterisks) */
    #[doc = ""]
    A,
      B,
);


foo!(
    /*** Only a comment */
    A,
      B,
);

foo!(
    #[doc = ""]
    /*** Only a comment */
    A,
      B,
);

foo!(
    /*** Only a comment */
    #[doc = ""]
    A,
      B,
);


// Inner doc comments cannot attach to a macro argument, so these are left
// unformatted on purpose.

foo!(
    //! Inner line doc
    A,
      B,
);

foo!(
    //!! Still an inner line doc (but with a bang at the beginning)
    A,
      B,
);

foo!(
    /*! Inner block doc */
    A,
      B,
);

foo!(
    /*!! Still an inner block doc (but with a bang at the beginning) */
    A,
      B,
);