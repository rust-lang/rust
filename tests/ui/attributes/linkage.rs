#![feature(linkage)]
#![feature(stmt_expr_attributes)]
#![deny(unused_attributes)]
#![allow(dead_code)]

#[linkage = "weak"] //~ ERROR attribute cannot be used on
type InvalidTy = ();

#[linkage = "weak"] //~ ERROR attribute cannot be used on
mod invalid_module {}

#[linkage = "weak"] //~ ERROR attribute cannot be used on
struct F;

#[linkage = "weak"] //~ ERROR attribute cannot be used on
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
    //~^^^^ ERROR attribute cannot be used on
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
    //~^^ ERROR attribute cannot be used on
}
