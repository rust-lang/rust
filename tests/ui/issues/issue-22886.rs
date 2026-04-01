// Regression test for #22886.

fn crash_please() {
    let mut iter = Newtype(Some(Box::new(0)));
    let saved = iter.next().unwrap();
    println!("{}", saved);
    iter.0 = None;
    println!("{}", saved);
}

struct Newtype(Option<Box<usize>>);

impl<'a> Iterator for Newtype { //~ ERROR E0207
    type Item = &'a Box<usize>;

    fn next(&mut self) -> Option<&Box<usize>> {
        self.0.as_ref()
    }
}

fn main() { }
