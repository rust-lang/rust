// @has manual_impl/trait.T.html
// @has  - '//*[@class="docblock"]' 'Docs associated with the trait definition.'
// @has  - '//*[@class="docblock"]' 'Docs associated with the trait a_method definition.'
// @has  - '//*[@class="docblock"]' 'Docs associated with the trait b_method definition.'
/// Docs associated with the trait definition.
pub trait T {
    /// Docs associated with the trait a_method definition.
    fn a_method(&self) -> usize;

    /// Docs associated with the trait b_method definition.
    fn b_method(&self) -> usize {
        self.a_method()
    }

    /// Docs associated with the trait c_method definition.
    ///
    /// There is another line
    fn c_method(&self) -> usize {
        self.a_method()
    }
}

// @has manual_impl/struct.S1.html '//*[@class="trait"]' 'T'
// @has  - '//*[@class="docblock"]' 'Docs associated with the S1 trait implementation.'
// @has  - '//*[@class="docblock"]' 'Docs associated with the S1 trait a_method implementation.'
// @!has - '//*[@class="docblock"]' 'Docs associated with the trait a_method definition.'
// @has - '//*[@class="docblock"]' 'Docs associated with the trait b_method definition.'
// @has - '//*[@class="docblock"]' 'Docs associated with the trait c_method definition.'
// @!has - '//*[@class="docblock"]' 'There is another line'
// @has - '//*[@class="docblock"]' 'Read more'
pub struct S1(usize);

/// Docs associated with the S1 trait implementation.
impl T for S1 {
    /// Docs associated with the S1 trait a_method implementation.
    fn a_method(&self) -> usize {
        self.0
    }
}

// @has manual_impl/struct.S2.html '//*[@class="trait"]' 'T'
// @has  - '//*[@class="docblock"]' 'Docs associated with the S2 trait implementation.'
// @has  - '//*[@class="docblock"]' 'Docs associated with the S2 trait a_method implementation.'
// @has  - '//*[@class="docblock"]' 'Docs associated with the S2 trait c_method implementation.'
// @!has - '//*[@class="docblock"]' 'Docs associated with the trait a_method definition.'
// @!has - '//*[@class="docblock"]' 'Docs associated with the trait c_method definition.'
// @has - '//*[@class="docblock"]' 'Docs associated with the trait b_method definition.'
pub struct S2(usize);

/// Docs associated with the S2 trait implementation.
impl T for S2 {
    /// Docs associated with the S2 trait a_method implementation.
    fn a_method(&self) -> usize {
        self.0
    }

    /// Docs associated with the S2 trait c_method implementation.
    fn c_method(&self) -> usize {
        5
    }
}

// @has manual_impl/struct.S3.html '//*[@class="trait"]' 'T'
// @has  - '//*[@class="docblock"]' 'Docs associated with the S3 trait implementation.'
// @has  - '//*[@class="docblock"]' 'Docs associated with the S3 trait b_method implementation.'
// @has - '//*[@class="docblock hidden"]' 'Docs associated with the trait a_method definition.'
pub struct S3(usize);

/// Docs associated with the S3 trait implementation.
impl T for S3 {
    fn a_method(&self) -> usize {
        self.0
    }

    /// Docs associated with the S3 trait b_method implementation.
    fn b_method(&self) -> usize {
        5
    }
}
