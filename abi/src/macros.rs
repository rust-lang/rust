//! Declarative macros for creating Things.

/// Create a Thing<T> from a struct literal.
///
/// Example:
/// ```ignore
/// let w = thing!(Window { x: 10, y: 20, ... });
/// ```
#[macro_export]
macro_rules! thing {
    ($val:expr) => {{
        $crate::Thing::new($val)
    }};
}

/// Create a Thing<Edge>.
///
/// Example:
/// ```ignore
/// let e = edge!(from, pred, to, flags);
/// ```
#[macro_export]
macro_rules! edge {
    ($from:expr, $pred:expr, $to:expr, $flags:expr $(,)?) => {{
        $crate::Thing::new($crate::graphable::Edge {
            from: $from,
            predicate: $pred,
            to: $to,
            flags: $flags,
        })
    }};
}
