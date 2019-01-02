/// Utility macro to test linting behavior in `option_methods()`
/// The lints included in `option_methods()` should not lint if the call to map is partially
/// within a macro
macro_rules! opt_map {
    ($opt:expr, $map:expr) => {($opt).map($map)};
}

/// Struct to generate false positive for Iterator-based lints
#[derive(Copy, Clone)]
struct IteratorFalsePositives {
    foo: u32,
}

impl IteratorFalsePositives {
    fn filter(self) -> IteratorFalsePositives {
        self
    }

    fn next(self) -> IteratorFalsePositives {
        self
    }

    fn find(self) -> Option<u32> {
        Some(self.foo)
    }

    fn position(self) -> Option<u32> {
        Some(self.foo)
    }

    fn rposition(self) -> Option<u32> {
        Some(self.foo)
    }

    fn nth(self, n: usize) -> Option<u32> {
        Some(self.foo)
    }

    fn skip(self, _: usize) -> IteratorFalsePositives {
        self
    }
}
