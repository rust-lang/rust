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
    let x = defer(&vec!["Goodbye", "world!"]); //~ ERROR temporary value dropped while borrowed
    x.x[0];
}
