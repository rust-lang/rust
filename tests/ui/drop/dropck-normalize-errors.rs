// Test that we don't ICE when computing the drop types for

trait Decode<'a> {
    type Decoder;
}

trait NonImplementedTrait {
    type Assoc;
}
struct NonImplementedStruct;

pub struct ADecoder<'a> {
    b: <B as Decode<'a>>::Decoder,
}
fn make_a_decoder<'a>() -> ADecoder<'a> {
    //~^ ERROR the trait bound
    //~| ERROR the trait bound
    panic!()
}

struct B;
impl<'a> Decode<'a> for B {
    type Decoder = BDecoder;
    //~^ ERROR the trait bound
}
pub struct BDecoder {
    non_implemented: <NonImplementedStruct as NonImplementedTrait>::Assoc,
    //~^ ERROR the trait bound
}

fn main() {}
