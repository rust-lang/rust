//@ build-fail
//@ compile-flags: -Copt-level=0

pub fn encode_num<Writer: ExampleWriter>(n: u32, mut writer: Writer) -> Result<(), Writer::Error> {
    if n > 15 {
        encode_num(n / 16, &mut writer)?;
        //~^ ERROR: reached the recursion limit while instantiating
    }
    Ok(())
}

pub trait ExampleWriter {
    type Error;
}

impl<'a, T: ExampleWriter> ExampleWriter for &'a mut T {
    type Error = T::Error;
}

struct Error;

impl ExampleWriter for Error {
    type Error = ();
}

fn main() {
    encode_num(69, &mut Error).unwrap();
}
