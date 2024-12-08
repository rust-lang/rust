//@ known-bug: #95134
//@ compile-flags: -Copt-level=0

pub fn encode_num<Writer: ExampleWriter>(n: u32, mut writer: Writer) -> Result<(), Writer::Error> {
    if n > 15 {
        encode_num(n / 16, &mut writer)?;
    }
    Ok(())
}

pub trait ExampleWriter {
    type Error;
}

impl<'a, T: ExampleWriter> ExampleWriter for &'a mut T {
    type Error = T::Error;
}

struct EmptyWriter;

impl ExampleWriter for EmptyWriter {
    type Error = ();
}

fn main() {
    encode_num(69, &mut EmptyWriter).unwrap();
}
