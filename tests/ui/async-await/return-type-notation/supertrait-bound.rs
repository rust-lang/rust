//@ check-pass

trait IntFactory {
    fn stream(&self) -> impl Iterator<Item = i32>;
}
trait SendIntFactory: IntFactory<stream(..): Send> + Send {}

fn main() {}
