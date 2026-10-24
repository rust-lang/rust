// Regression test for issue #74918
// Tests that we don't ICE after emitting an error

struct ChunkingIterator<T, S: 'static + Iterator<Item = T>> {
    source: S,
}

impl<T, S: Iterator<Item = T>> Iterator for ChunkingIterator<T, S> {
    type Item = IteratorChunk<T, S>; //~ ERROR missing lifetime

    fn next(&mut self) -> Option<IteratorChunk<T, S>> {
        todo!()
    }
}

struct IteratorChunk<'a, T, S: Iterator<Item = T>> {
    source: &'a mut S,
}

impl<T, S: Iterator<Item = T>> Iterator for IteratorChunk<'_, T, S> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        todo!()
    }
}

fn main() {}
