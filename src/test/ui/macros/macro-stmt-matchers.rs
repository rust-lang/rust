#![feature(rustc_attrs)]

#[rustc_error]
fn main() { //~ ERROR compilation successful
    macro_rules! m { ($s:stmt;) => { $s } }
    m!(vec![].push(0););
}
