// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Hash<H> {
    fn hash2(&self, hasher: &H) -> u64;
}

trait Stream {
    fn input(&mut self, bytes: &[u8]);
    fn result(&self) -> u64;
}

trait StreamHasher<S: Stream> {
    fn stream(&self) -> S;
}

//////////////////////////////////////////////////////////////////////////////

trait StreamHash<S: Stream, H: StreamHasher<S>>: Hash<H> {
    fn input_stream(&self, stream: &mut S);
}

impl<S: Stream, H: StreamHasher<S>> Hash<H> for u8 {
    fn hash2(&self, hasher: &H) -> u64 {
        let mut stream = hasher.stream();
        self.input_stream(&mut stream);
        stream.result()
    }
}

impl<S: Stream, H: StreamHasher<S>> StreamHash<S, H> for u8 {
    fn input_stream(&self, stream: &mut S) {
        stream.input(&[*self]);
    }
}

fn main() {}
