//@ build-pass
//@ edition:2018

#![feature(coroutines, stmt_expr_attributes)]

fn main() {
    let _ = #[coroutine] static |x: u8| match x {
        y if { yield } == y + 1 => (),
        _ => (),
    };
}
