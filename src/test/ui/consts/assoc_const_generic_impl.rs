#![warn(const_err)]

trait ZeroSized: Sized {
    const I_AM_ZERO_SIZED: ();
    fn requires_zero_size(self);
}

impl<T: Sized> ZeroSized for T {
    const I_AM_ZERO_SIZED: ()  = [()][std::mem::size_of::<Self>()]; //~ WARN any use of this value
    fn requires_zero_size(self) {
        let () = Self::I_AM_ZERO_SIZED; //~ ERROR erroneous constant encountered
        println!("requires_zero_size called");
    }
}

fn main() {
    ().requires_zero_size();
    42_u32.requires_zero_size();
}
