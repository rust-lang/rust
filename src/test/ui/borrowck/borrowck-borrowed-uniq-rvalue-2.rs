// error-pattern: borrowed value does not live long enough

struct defer<'a> {
    x: &'a [&'a str],
}

impl<'a> Drop for defer<'a> {
    fn drop(&mut self) {
        unsafe {
            println!("{:?}", self.x);
        }
    }
}

fn defer<'r>(x: &'r [&'r str]) -> defer<'r> {
    defer {
        x: x
    }
}

fn main() {
    let x = defer(&vec!["Goodbye", "world!"]);
    x.x[0];
}
