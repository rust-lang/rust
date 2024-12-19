pub enum En {
    A(Vec<u8>)
}

fn get_usize() -> usize {
    0
}

macro_rules! force_expr {
    ($e:expr) => { $e } //~ ERROR expected pattern, found expression `Vec :: new()`
}

macro_rules! force_pat {
    ($a:expr, $b:expr) => { $a..=$b } //~ ERROR expected pattern, found expression `get_usize()`
}

macro_rules! make_vec {
    () => { force_expr!(Vec::new()) }
}

macro_rules! make_pat {
    () => { force_pat!(get_usize(), get_usize()) }
}

#[allow(unreachable_code)]
fn f() -> Result<(), impl core::fmt::Debug> {
    let x: En = loop {};

    assert!(matches!(x, En::A(make_vec!())));
    assert!(matches!(5, make_pat!()));
    Ok::<(), &'static str>(())
}

fn main() {}
