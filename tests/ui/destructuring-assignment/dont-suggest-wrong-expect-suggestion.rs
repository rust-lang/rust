//! regression test for <https://github.com/rust-lang/rust/issues/155516>

struct NamedStruct {
    opt_field: Option<usize>,
    res_field: Result<usize, ()>,
}

fn main() {
    let a: usize;
    let b: usize;

    NamedStruct {
        opt_field: a, //~ ERROR mismatched types
        res_field: b, //~ ERROR mismatched types
    } = NamedStruct {
        opt_field: Some(0),
        res_field: Ok(42_usize),
    };
}
