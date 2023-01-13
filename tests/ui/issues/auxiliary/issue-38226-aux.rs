#![crate_type="rlib"]

#[inline(never)]
pub fn foo<T>() {
    let _: Box<SomeTrait> = Box::new(SomeTraitImpl);
}

pub fn bar() {
    SomeTraitImpl.bar();
}

mod submod {
    pub trait SomeTrait {
        fn bar(&self) {
            panic!("NO")
        }
    }
}

use self::submod::SomeTrait;

pub struct SomeTraitImpl;
impl SomeTrait for SomeTraitImpl {}
