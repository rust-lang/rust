//@ build-fail
fn main() {
    encode(&mut EncoderImpl);
}

pub trait Encoder {
    type W;

    fn writer(&self) -> Self::W;
}

fn encode<E: Encoder>(mut encoder: E) {
//~^ WARN: function cannot return without recursing
    encoder.writer();
    encode(&mut encoder);
    //~^ ERROR: reached the recursion limit while instantiating
}

struct EncoderImpl;

impl Encoder for EncoderImpl {
    type W = ();

    fn writer(&self) {}
}

impl<'a, T: Encoder> Encoder for &'a mut T {
    type W = T::W;

    fn writer(&self) -> Self::W {
        panic!()
    }
}
