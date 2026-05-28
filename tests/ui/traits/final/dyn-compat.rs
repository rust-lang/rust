//@ run-pass

#![feature(final_associated_functions)]

trait FinalNoReceiver {
    final fn no_receiver() {}
}

trait FinalGeneric {
    final fn generic<T>(&self, _value: T) {}
}

trait FinalSelfParam {
    final fn self_param(&self, _other: &Self) {}
}

trait FinalSelfReturn {
    final fn self_return(&self) -> &Self {
        self
    }
}

struct S;

impl FinalNoReceiver for S {}
impl FinalGeneric for S {}
impl FinalSelfParam for S {}
impl FinalSelfReturn for S {}

fn main() {
    let s = S;
    <S as FinalNoReceiver>::no_receiver();
    let obj_generic: &dyn FinalGeneric = &s;
    let obj_param: &dyn FinalSelfParam = &s;
    let obj_return: &dyn FinalSelfReturn = &s;
    obj_generic.generic(1u8);
    obj_param.self_param(obj_param);
    let _ = obj_return.self_return();
    let _: &dyn FinalNoReceiver = &s;
}
