use std::fmt;

// Convenience macros for writing and unwrapping.
#[macro_export]
macro_rules! writeln {
    ($($args:tt)*) => {{
        use std::fmt::Write as _;
        std::writeln!($($args)*).unwrap();
    }};
}
#[macro_export]
macro_rules! write {
    ($($args:tt)*) => {{
        use std::fmt::Write as _;
        std::write!($($args)*).unwrap();
    }};
}

pub fn fmt_list<V: fmt::Debug>(values: impl IntoIterator<Item = V>) -> String {
    let pieces = values.into_iter().map(|b| format!("{b:?}, "));
    let mut out = String::new();
    let mut line = String::from("\n    ");
    for piece in pieces {
        if line.len() + piece.len() < 98 {
            line.push_str(&piece);
        } else {
            writeln!(out, "{}", line.trim_end());
            line = format!("    {piece}");
        }
    }
    writeln!(out, "{}", line.trim_end());
    out
}

/// Wrapper type for formatting a `T` using its `Binary` implementation.
#[derive(Copy, Clone)]
pub struct Bin<T>(pub T);

impl<T: fmt::Binary> fmt::Debug for Bin<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bits = size_of::<T>() * 8;
        std::write!(f, "0b{:0bits$b}", self.0)
    }
}

impl<T: fmt::Binary> fmt::Display for Bin<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Wrapper type for formatting a `T` using its `LowerHex` implementation.
#[derive(Copy, Clone)]
pub struct Hex<T>(pub T);

impl<T: fmt::LowerHex> fmt::Debug for Hex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        std::write!(f, "{:#x}", self.0)
    }
}

impl<T: fmt::LowerHex> fmt::Display for Hex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Wrapper type for formatting a `char` using `escape_unicode`.
#[derive(Copy, Clone)]
pub struct CharEscape(pub char);

impl fmt::Debug for CharEscape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        std::write!(f, "'{}'", self.0.escape_unicode())
    }
}

impl fmt::Display for CharEscape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}
