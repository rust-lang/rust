//@ check-pass
//
// A test case where a `block` fragment specifier is interpreted as an `expr`
// fragment specifier. It's an interesting case for the handling of invisible
// delimiters.

macro_rules! m_expr {
    ($e:expr) => { const _CURRENT: u32 = $e; };
}

macro_rules! m_block {
    ($b:block) => ( m_expr!($b); );
}

fn main() {
    m_block!({ 1 });
}
