// compile-flags: -Z unstable-options

#![feature(rustc_attrs)]
#![forbid(rustc::ty_compare_operator)]

mod ty1 {
    #[derive(PartialEq, PartialOrd)]
    #[rustc_diagnostic_item = "Ty"]
    pub struct Ty;
}

mod ty2 {
    #[derive(PartialEq, PartialOrd)]
    pub struct Ty;
}

fn main() {
    let _ = ty1::Ty == ty1::Ty;
    //~^ ERROR using the a comparison operator on `Ty`
    let _ = ty1::Ty != ty1::Ty;
    //~^ ERROR using the a comparison operator on `Ty`
    let _ = ty1::Ty < ty1::Ty;
    //~^ ERROR using the a comparison operator on `Ty`
    let _ = ty1::Ty <= ty1::Ty;
    //~^ ERROR using the a comparison operator on `Ty`
    let _ = ty1::Ty > ty1::Ty;
    //~^ ERROR using the a comparison operator on `Ty`
    let _ = ty1::Ty >= ty1::Ty;
    //~^ ERROR using the a comparison operator on `Ty`

    let _ = ty2::Ty == ty2::Ty;
    let _ = ty2::Ty != ty2::Ty;
    let _ = ty2::Ty < ty2::Ty;
    let _ = ty2::Ty <= ty2::Ty;
    let _ = ty2::Ty > ty2::Ty;
    let _ = ty2::Ty >= ty2::Ty;
}
