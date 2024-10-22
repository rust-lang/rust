//@ known-bug: #131648
#![feature(return_type_notation)]

trait IntFactory {
    fn stream(self) -> impl IntFactory<stream(..): Send>;
}
fn main() {}
