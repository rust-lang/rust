#![feature(const_trait_impl)]

pub trait MyTrait {
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
