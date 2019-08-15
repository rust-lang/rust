#![feature(associated_type_bounds)]

fn main() {}

trait Bar { type Assoc; }

trait Thing {
    type Out;
    fn func() -> Self::Out;
}

struct AssocNoCopy;
impl Bar for AssocNoCopy { type Assoc = String; }

impl Thing for AssocNoCopy {
    type Out = Box<dyn Bar<Assoc: Copy>>;
    //~^ ERROR the trait bound `std::string::String: std::marker::Copy` is not satisfied

    fn func() -> Self::Out {
        Box::new(AssocNoCopy)
    }
}
