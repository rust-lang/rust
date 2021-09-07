#![warn(clippy::iter_not_returning_iterator)]

struct Data {
    begin: u32,
}

struct Counter {
    count: u32,
}

impl Data {
    fn iter(&self) -> Counter {
        todo!()
    }

    fn iter_mut(&self) -> Counter {
        todo!()
    }
}

struct Data2 {
    begin: u32,
}

struct Counter2 {
    count: u32,
}

impl Data2 {
    fn iter(&self) -> Counter2 {
        todo!()
    }

    fn iter_mut(&self) -> Counter2 {
        todo!()
    }
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

fn main() {}
