trait T<'x> {
    type V;
}

impl<'g> T<'g> for u32 {
    type V = u16;
}

fn main() {
    (&|_| ()) as &dyn for<'x> Fn(<u32 as T<'x>>::V);
    //~^ ERROR: type mismatch in closure arguments
    //~| ERROR: size for values of type `<u32 as T<'_>>::V` cannot be known at compilation time
}
