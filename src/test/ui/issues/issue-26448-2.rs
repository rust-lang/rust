// run-pass

pub struct Bar<T> {
    items: Vec<&'static str>,
    inner: T,
}

pub trait IntoBar<T> {
    fn into_bar(self) -> Bar<T>;
}

impl<'a, T> IntoBar<T> for &'a str where &'a str: Into<T> {
    fn into_bar(self) -> Bar<T> {
        Bar {
            items: Vec::new(),
            inner: self.into(),
        }
    }
}

fn main() {}
