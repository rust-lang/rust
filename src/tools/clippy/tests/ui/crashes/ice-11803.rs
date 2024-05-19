//@no-rustfix

#![warn(clippy::impl_trait_in_params)]

pub fn g<T: IntoIterator<Item = impl Iterator<Item = impl Clone>>>() {
    extern "C" fn implementation_detail() {}
}

fn main() {}
