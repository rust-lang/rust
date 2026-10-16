#![feature(return_type_notation)]

trait IntFactory {
    fn stream(&self) -> impl IntFactory<stream(..): Send>;
}

struct S;

impl IntFactory for S {
    fn stream(&self) -> impl IntFactory<stream(..): Send> {
        //~^ ERROR the trait bound `!: IntFactory` is not satisfied
        todo!()
    }
}

fn main() {}
