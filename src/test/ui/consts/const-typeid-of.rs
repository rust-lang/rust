use std::any::TypeId;

struct A;

fn main() {
    const A_ID: TypeId = TypeId::of::<A>();
    //~^ ERROR `std::any::TypeId::of` is not yet stable as a const fn
}
