//@ build-fail

trait ZeroSized: Sized {
    const I_AM_ZERO_SIZED: ();
    fn requires_zero_size(self);
}

impl<T: Sized> ZeroSized for T {
    const I_AM_ZERO_SIZED: () = [()][std::mem::size_of::<Self>()]; //~ ERROR index out of bounds: the length is 1 but the index is 4
    fn requires_zero_size(self) {
        Self::I_AM_ZERO_SIZED;
        println!("requires_zero_size called");
    }
}

fn main() {
    ().requires_zero_size();
    42_u32.requires_zero_size();
}
