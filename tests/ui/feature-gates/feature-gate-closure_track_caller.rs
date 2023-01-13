#![feature(stmt_expr_attributes)]
#![feature(generators)]

fn main() {
    let _closure = #[track_caller] || {}; //~ `#[track_caller]` on closures
    let _generator = #[track_caller] || { yield; }; //~ `#[track_caller]` on closures
}
