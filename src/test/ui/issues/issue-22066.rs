// build-pass (FIXME(62277): could be check-pass?)
pub trait LineFormatter<'a> {
    type Iter: Iterator<Item=&'a str> + 'a;
    fn iter(&'a self, line: &'a str) -> Self::Iter;

    fn dimensions(&'a self, line: &'a str) {
        let iter: Self::Iter = self.iter(line);
        <_ as IntoIterator>::into_iter(iter);
    }
}

fn main() {}
