// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

pub struct MyArray<const COUNT: usize>([u8; COUNT + 1]);
//[full]~^ ERROR constant expression depends on a generic parameter
//[min]~^^ ERROR generic parameters may not be used

impl<const COUNT: usize> MyArray<COUNT> {
    fn inner(&self) -> &[u8; COUNT + 1] {
        //[full]~^ ERROR constant expression depends on a generic parameter
        //[min]~^^ ERROR generic parameters may not be used
        &self.0
    }
}

fn main() {}
