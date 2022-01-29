fn iterate<T, F>(initial: T, f: F) -> Iterate<T, F> {
    Iterate {
        state: initial,
        f: f,
    }
}

pub struct Iterate<T, F> {
    state: T,
    f: F
}

impl<T: Clone, F> Iterator for Iterate<T, F> where F: Fn(&T) -> T {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.state = (self.f)(&self.state);
        Some(self.state.clone())
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { (usize::MAX, None) }
}

fn main() {
    let a = iterate(0, |x| x+1);
    println!("{:?}", a.iter().take(10).collect::<Vec<usize>>());
    //~^ ERROR no method named `iter` found
}
