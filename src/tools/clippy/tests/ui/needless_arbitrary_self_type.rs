#![warn(clippy::needless_arbitrary_self_type)]
#![allow(unused_mut, clippy::needless_lifetimes)]

pub enum ValType {
    A,
    B,
}

impl ValType {
    pub fn bad(self: Self) {
        //~^ needless_arbitrary_self_type
        unimplemented!();
    }

    pub fn good(self) {
        unimplemented!();
    }

    pub fn mut_bad(mut self: Self) {
        //~^ needless_arbitrary_self_type
        unimplemented!();
    }

    pub fn mut_good(mut self) {
        unimplemented!();
    }

    pub fn ref_bad(self: &Self) {
        //~^ needless_arbitrary_self_type
        unimplemented!();
    }

    pub fn ref_good(&self) {
        unimplemented!();
    }

    pub fn ref_bad_with_lifetime<'a>(self: &'a Self) {
        //~^ needless_arbitrary_self_type
        unimplemented!();
    }

    pub fn ref_good_with_lifetime<'a>(&'a self) {
        unimplemented!();
    }

    pub fn mut_ref_bad(self: &mut Self) {
        //~^ needless_arbitrary_self_type
        unimplemented!();
    }

    pub fn mut_ref_good(&mut self) {
        unimplemented!();
    }

    pub fn mut_ref_bad_with_lifetime<'a>(self: &'a mut Self) {
        //~^ needless_arbitrary_self_type
        unimplemented!();
    }

    pub fn mut_ref_good_with_lifetime<'a>(&'a mut self) {
        unimplemented!();
    }

    pub fn mut_ref_mut_good(mut self: &mut Self) {
        unimplemented!();
    }

    pub fn mut_ref_mut_ref_good(self: &&mut &mut Self) {
        unimplemented!();
    }
}

trait Foo<'r#struct> {
    fn f1(self: &'r#struct Self) {}
    //~^ needless_arbitrary_self_type
    fn f2(self: &'r#struct mut Self) {}
    //~^ needless_arbitrary_self_type
}

fn main() {}
