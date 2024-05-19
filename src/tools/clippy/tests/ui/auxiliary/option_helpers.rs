#![allow(dead_code, unused_variables, clippy::return_self_not_must_use)]

/// Utility macro to test linting behavior in `option_methods()`
/// The lints included in `option_methods()` should not lint if the call to map is partially
/// within a macro
#[macro_export]
macro_rules! opt_map {
    ($opt:expr, $map:expr) => {
        ($opt).map($map)
    };
}

/// Struct to generate false positive for Iterator-based lints
#[derive(Copy, Clone)]
pub struct IteratorFalsePositives {
    pub foo: u32,
}

impl IteratorFalsePositives {
    pub fn filter(self) -> IteratorFalsePositives {
        self
    }

    pub fn next(self) -> IteratorFalsePositives {
        self
    }

    pub fn find(self) -> Option<u32> {
        Some(self.foo)
    }

    pub fn position(self) -> Option<u32> {
        Some(self.foo)
    }

    pub fn rposition(self) -> Option<u32> {
        Some(self.foo)
    }

    pub fn nth(self, n: usize) -> Option<u32> {
        Some(self.foo)
    }

    pub fn skip(self, _: usize) -> IteratorFalsePositives {
        self
    }

    pub fn skip_while(self) -> IteratorFalsePositives {
        self
    }

    pub fn count(self) -> usize {
        self.foo as usize
    }
}

#[derive(Copy, Clone)]
pub struct IteratorMethodFalsePositives;

impl IteratorMethodFalsePositives {
    pub fn filter(&self, _s: i32) -> std::vec::IntoIter<i32> {
        unimplemented!();
    }
}
