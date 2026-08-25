//@ check-pass

use std::fmt::Debug;

pub struct EventStream<S> {
    stream: S,
}

impl<S: Debug> EventStream<S> {
    fn into_stream(self) -> impl Debug {
        unimplemented!()
    }

    pub fn into_reader(self) -> impl Debug {
        ReaderStream::from(self.into_stream())
    }
}

#[derive(Debug)]
pub struct ReaderStream<S> {
    stream: S,
}

impl<S> From<S> for ReaderStream<S> {
    fn from(stream: S) -> Self {
        ReaderStream { stream }
    }
}

fn main() {}
