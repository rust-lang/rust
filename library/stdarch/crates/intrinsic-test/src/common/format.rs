//! Basic code formatting tools.
//!
//! We don't need perfect formatting for the generated tests, but simple indentation can make
//! debugging a lot easier.

#[derive(Copy, Clone, Debug, Default)]
pub struct Indentation(u32);

impl Indentation {
    pub fn nested(self) -> Self {
        Self(self.0 + 1)
    }
}

impl std::fmt::Display for Indentation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for _ in 0..self.0 {
            write!(f, "    ")?;
        }
        Ok(())
    }
}
