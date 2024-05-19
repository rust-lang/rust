//@ known-bug: #92470
fn main() {
    encode(&mut EncoderImpl);
}

pub trait Encoder {
    type W;

    fn writer(&self) -> Self::W;
}

fn encode<E: Encoder>(mut encoder: E) {
    encoder.writer();
    encode(&mut encoder);
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
