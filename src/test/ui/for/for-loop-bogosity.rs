struct MyStruct {
    x: isize,
    y: isize,
}

impl MyStruct {
    fn next(&mut self) -> Option<isize> {
        Some(self.x)
    }
}

pub fn main() {
    let mut bogus = MyStruct {
        x: 1,
        y: 2,
    };
    for x in bogus { //~ ERROR `MyStruct: std::iter::Iterator` is not satisfied
        drop(x);
    }
}
