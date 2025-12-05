fn ignore<F>(_f: F) where F: for<'z> FnOnce(&'z isize) -> &'z isize {}

fn nested() {
    let y = 3;
    ignore(
        |z| {
            if false { &y } else { z }
            //~^ ERROR `y` does not live long enough
        });
}

fn main() {}
