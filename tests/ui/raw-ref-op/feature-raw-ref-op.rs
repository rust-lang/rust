// gate-test-raw_ref_op

macro_rules! is_expr {
    ($e:expr) => {}
}

is_expr!(&raw const a);         //~ ERROR raw address of syntax is experimental
is_expr!(&raw mut a);           //~ ERROR raw address of syntax is experimental

#[cfg(FALSE)]
fn cfgd_out() {
    let mut a = 0;
    &raw const a;               //~ ERROR raw address of syntax is experimental
    &raw mut a;                 //~ ERROR raw address of syntax is experimental
}

fn main() {
    let mut y = 123;
    let x = &raw const y;       //~ ERROR raw address of syntax is experimental
    let x = &raw mut y;         //~ ERROR raw address of syntax is experimental
}
