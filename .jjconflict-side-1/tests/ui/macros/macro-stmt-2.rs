//@ check-pass
//
// This shows a tricky case for #124141, where `declare!(_x)` was incorrectly
// being categorised as a `StmtKind::Expr` instead of a `StmtKind::MacCall` in
// `parse_stmt_mac`.

macro_rules! as_stmt { ($s:stmt) => { $s }; }

macro_rules! declare { ($name:ident) => { let $name = 0u32; }; }

fn main() { as_stmt!(declare!(_x)); }
