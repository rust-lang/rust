// rustfmt-wrap_comments: true
// Test attributes and doc comments are preserved.

/// Blah blah blah.
/// Blah blah blah.
/// Blah blah blah.
/// Blah blah blah.

/// Blah blah blah.
impl Bar {
    /// Blah blah blooo.
    /// Blah blah blooo.
    /// Blah blah blooo.
    /// Blah blah blooo.
    #[an_attribute]
    fn foo(&mut self) -> isize {
    }

    /// Blah blah bing.
    /// Blah blah bing.
    /// Blah blah bing.

    /// Blah blah bing.
    /// Blah blah bing.
    /// Blah blah bing.
    pub fn f2(self) {
        (foo, bar)
    }

    #[another_attribute]
    fn f3(self) -> Dog {
    }

    /// Blah blah bing.
    #[attrib1]
    /// Blah blah bing.
    #[attrib2]
    // Another comment that needs rewrite because it's tooooooooooooooooooooooooooooooo
    // loooooooooooong.
    /// Blah blah bing.
    fn f4(self) -> Cat {
    }
}
