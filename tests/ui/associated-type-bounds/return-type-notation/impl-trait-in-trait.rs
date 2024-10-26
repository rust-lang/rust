#![feature(return_type_notation)]

trait IntFactory {
    fn stream(self) -> impl IntFactory<stream(..): Send>;
}

fn main() {}
