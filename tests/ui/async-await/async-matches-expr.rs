//@ build-pass (FIXME(62277): could be check-pass?)
//@ edition:2018

macro_rules! match_expr {
    ($x:expr) => {}
}

fn main() {
    match_expr!(async {});
}
