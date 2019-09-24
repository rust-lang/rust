//
// build-pass (FIXME(62277): could be check-pass?)

trait DictLike<'a> {
    type ItemsIterator: Iterator<Item=u8>;
    fn get(c: Self::ItemsIterator) {
        c.into_iter();
    }
}

trait DictLike2<'a> {
    type ItemsIterator: Iterator<Item=u8>;

    fn items(&self) -> Self::ItemsIterator;

    fn get(&self)  {
        for _ in self.items() {}
    }
}

fn main() {}
