struct Defer<'a> {
    x: &'a [&'a str],
}

impl<'a> Drop for Defer<'a> {
    fn drop(&mut self) {
        unsafe {
            println!("{:?}", self.x);
        }
    }
}

fn defer<'r>(x: &'r [&'r str]) -> Defer<'r> {
    Defer {
        x: x
    }
}

fn main() {
    let x = defer(&vec!["Goodbye", "world!"]); //~ ERROR borrowed value does not live long enough
    x.x[0];
}
