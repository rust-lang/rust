#![feature(return_type_notation)]

trait IntFactory {
    fn stream(&self) -> impl IntFactory<stream(..): Send>;
    //~^ ERROR recursion limit exceeded while collecting implied RPITIT bounds
}

struct S;

impl IntFactory for S {
    fn stream(&self) -> impl IntFactory<stream(..): Send> {
        //~^ ERROR the trait bound `!: IntFactory` is not satisfied
        todo!()
    }
}

fn main() {}
