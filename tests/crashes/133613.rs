//@ known-bug: #133613

struct Wrapper<'a>();

trait IntFactory {
    fn stream(&self) -> impl IntFactory<stream(..): IntFactory<stream(..): Send>>;
}
