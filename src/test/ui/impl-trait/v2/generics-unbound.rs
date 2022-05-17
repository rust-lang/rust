// check-pass

#![feature(return_position_impl_trait_v2)]

trait MyTrait<T> {
    fn get_inner(self) -> T;
}

impl<T> MyTrait<T> for T {
    fn get_inner(self) -> Self {
        self
    }
}

fn ident_as_my_trait<T>(t: T) -> impl MyTrait<T> {
    t
}

fn main() {
    assert_eq!(22, ident_as_my_trait(22).get_inner());
}
