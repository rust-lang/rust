#![feature(linkage)]
#![feature(stmt_expr_attributes)]
#![deny(unused_attributes)]
#![allow(dead_code)]

#[linkage = "weak"] //~ ERROR attribute should be applied to a function or static
type InvalidTy = ();

#[linkage = "weak"] //~ ERROR attribute should be applied to a function or static
mod invalid_module {}

#[linkage = "weak"] //~ ERROR attribute should be applied to a function or static
struct F;

#[linkage = "weak"] //~ ERROR attribute should be applied to a function or static
impl F {
    #[linkage = "weak"]
    fn valid(&self) {}
}

#[linkage = "weak"]
fn f() {
    #[linkage = "weak"]
    {
        1
    };
    //~^^^^ ERROR attribute should be applied to a function or static
}

extern "C" {
    #[linkage = "weak"]
    static A: *const ();

    #[linkage = "weak"]
    fn bar();
}

fn main() {
    let _ = #[linkage = "weak"]
    (|| 1);
    //~^^ ERROR attribute should be applied to a function or static
}
