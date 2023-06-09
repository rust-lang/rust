#![crate_name = "foo"]

// @has foo/trait.Read.html
// @has - '//h2' 'Trait examples'
/// # Trait examples
pub trait Read {
    // @has - '//h5' 'Function examples'
    /// # Function examples
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ()>;
}

pub struct Foo;

// @has foo/struct.Foo.html
impl Foo {
    // @has - '//h5' 'Implementation header'
    /// # Implementation header
    pub fn bar(&self) -> usize {
        1
    }
}

impl Read for Foo {
    // @has - '//h5' 'Trait implementation header'
    /// # Trait implementation header
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ()> {
        Ok(1)
    }
}
