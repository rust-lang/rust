//@ run-pass

pub trait Iter {
    type Item<'a> where Self: 'a;

    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>>;

    fn for_each<F>(mut self, mut f: F)
        where Self: Sized, F: for<'a> FnMut(Self::Item<'a>)
    {
        while let Some(item) = self.next() {
            f(item);
        }
    }
}

pub struct Windows<T> {
    items: Vec<T>,
    start: usize,
    len: usize,
}

impl<T> Windows<T> {
    pub fn new(items: Vec<T>, len: usize) -> Self {
        Self { items, start: 0, len }
    }
}

impl<T> Iter for Windows<T> {
    type Item<'a> = &'a mut [T] where T: 'a;

    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>> {
        let slice = self.items.get_mut(self.start..self.start + self.len)?;
        self.start += 1;
        Some(slice)
    }
}

fn main() {
    Windows::new(vec![1, 2, 3, 4, 5], 3)
        .for_each(|slice| println!("{:?}", slice));
}
