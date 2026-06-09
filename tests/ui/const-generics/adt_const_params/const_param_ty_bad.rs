#![allow(incomplete_features)]
#![feature(adt_const_params, unsized_const_params)]

fn check(_: impl std::marker::ConstParamTy_) {}

fn main() {
    check(main); //~ error: `fn() {main}` can't be used as a const parameter type
    check(|| {}); //~ error: `{closure@$DIR/const_param_ty_bad.rs:8:11: 8:13}` can't be used as a const parameter type
    check(main as fn()); //~ error: `fn()` can't be used as a const parameter type
    check(&mut ()); //~ error: `&mut ()` can't be used as a const parameter type
    check(&mut () as *mut ()); //~ error: `*mut ()` can't be used as a const parameter type
    check(&() as *const ()); //~ error: `*const ()` can't be used as a const parameter type
}
