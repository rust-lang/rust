#![rustfmt::skip::attributes(skip_mod_attr)]

mod sub_mod;

#[rustfmt::skip::attributes(other, skip_attr)]
fn main() {
    #[other(should,
skip,
        this,                               format)]
    struct S {}

    #[skip_attr(should, skip,
this,                               format,too)]
    fn doesnt_mater() {}

    #[skip_mod_attr(should, skip,
this,                               format,
         enerywhere)]
    fn more() {}

    #[not_skip(not, skip, me)]
    struct B {}
}

#[other(should, not, skip, this, format, here)]
fn foo() {}

#[skip_mod_attr(should, skip,
this,                               format,in,                    master,
                    and, sub, module)]
fn bar() {}
