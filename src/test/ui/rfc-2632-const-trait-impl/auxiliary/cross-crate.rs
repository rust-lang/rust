#![feature(const_fn_trait_bound)]
#![feature(const_trait_impl)]

pub trait MyTrait {
    #[default_method_body_is_const]
    fn defaulted_func(&self) {}
    fn func(self);
}

pub struct NonConst;

impl MyTrait for NonConst {
    fn func(self) {

    }
}

pub struct Const;

impl const MyTrait for Const {
    fn func(self) {

    }
}
