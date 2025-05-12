//@ build-pass

pub trait A {}
pub trait B {}

pub trait C: A + B {}
impl<X: A + B> C for X {}

pub fn test<'a, T>(view: T) -> Option<&'a mut dyn B>
where
    T: IntoIterator<Item = &'a mut dyn B>,
{
    return Some(view.into_iter().next().unwrap());
}

fn main() {
    let mut a: Vec<Box<dyn C>> = Vec::new();
    test(a.iter_mut().map(|c| c.as_mut() as &mut dyn B));
}
