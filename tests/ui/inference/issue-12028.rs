// Test an example where we fail to infer the type parameter H. This
// is because there is really nothing constraining it. At one time, we
// would infer based on the where clauses in scope, but that no longer
// works.

trait Hash<H> {
    fn hash2(&self, hasher: &H) -> u64;
}

trait Stream {
    fn input(&mut self, bytes: &[u8]);
    fn result(&self) -> u64;
}

trait StreamHasher {
    type S : Stream;
    fn stream(&self) -> Self::S;
}

trait StreamHash<H: StreamHasher>: Hash<H> {
    fn input_stream(&self, stream: &mut H::S);
}

impl<H: StreamHasher> Hash<H> for u8 {
    fn hash2(&self, hasher: &H) -> u64 {
        let mut stream = hasher.stream();
        self.input_stream(&mut stream); //~ ERROR type annotations needed
        Stream::result(&stream)
    }
}

impl<H: StreamHasher> StreamHash<H> for u8 {
    fn input_stream(&self, stream: &mut H::S) {
        Stream::input(stream, &[*self]);
    }
}

fn main() {}
