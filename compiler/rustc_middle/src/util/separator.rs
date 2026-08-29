use std::fmt;

/// Interpose a separator between elements being printed
pub(crate) struct SeparatorPrinter {
    /// Separator to interpose
    sep: &'static str,
    /// If this is the first element
    first: bool,
}

impl SeparatorPrinter {
    /// Create a new printer
    pub(crate) fn new(sep: &'static str) -> Self {
        SeparatorPrinter { sep, first: true }
    }

    /// Print the separator if this is not the first element
    pub(crate) fn print_separator(&mut self, w: &mut impl fmt::Write) -> fmt::Result {
        if !self.first {
            w.write_str(self.sep)?;
        }
        self.first = false;
        Ok(())
    }

    /// Check if no element was printed before
    pub(crate) fn nothing_printed(&self) -> bool {
        self.first
    }
}
