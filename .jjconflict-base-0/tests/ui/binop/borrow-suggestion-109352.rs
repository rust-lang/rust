//@ run-rustfix

struct Foo;

impl std::ops::Mul for &Foo {
    type Output = Foo;

    fn mul(self, _rhs: Self) -> Self::Output {
        unimplemented!()
    }
}

fn main() {
    let ref_mut_foo: &mut Foo = &mut Foo;
    let ref_foo: &Foo = &Foo;
    let owned_foo: Foo = Foo;

    let _ = ref_foo * ref_foo;
    let _ = ref_foo * ref_mut_foo;

    let _ = ref_mut_foo * ref_foo;
    //~^ ERROR cannot multiply
    let _ = ref_mut_foo * ref_mut_foo;
    //~^ ERROR cannot multiply
    let _ = ref_mut_foo * &owned_foo;
    //~^ ERROR cannot multiply
}
