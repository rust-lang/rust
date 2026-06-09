// This was triggering an assertion failure in `NodeRange::new`.

//@ check-pass

#![feature(cfg_eval)]
#![feature(stmt_expr_attributes)]

fn f() -> u32 {
    #[cfg_eval] #[cfg(not(FALSE))] 0
}

fn main() {}
