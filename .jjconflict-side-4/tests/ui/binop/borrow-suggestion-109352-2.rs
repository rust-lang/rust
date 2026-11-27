struct Bar;

impl std::ops::Mul for &mut Bar {
    type Output = Bar;

    fn mul(self, _rhs: Self) -> Self::Output {
        unimplemented!()
    }
}

fn main() {
    let ref_mut_bar: &mut Bar = &mut Bar;
    let ref_bar: &Bar = &Bar;
    let owned_bar: Bar = Bar;

    let _ = ref_mut_bar * ref_mut_bar;

    // FIXME: we should be able to suggest borrowing both side
    let _ = owned_bar * owned_bar;
    //~^ ERROR cannot multiply
    let _ = ref_bar * ref_bar;
    //~^ ERROR cannot multiply
    let _ = ref_bar * ref_mut_bar;
    //~^ ERROR cannot multiply
    let _ = ref_mut_bar * ref_bar;
    //~^ ERROR mismatched types
}
