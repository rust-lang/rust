#![feature(return_type_notation)]

trait IntFactory {
    fn stream(&self) -> impl IntFactory<stream(..): Send>;
    //~^ ERROR recursion limit exceeded while collecting implied RPITIT bounds
}

impl IntFactory for u32 {
    fn stream(&self) -> Self {
        todo!()
    }
}

fn main() {}
