fn ignore<F>(_f: F) where F: for<'z> FnOnce(&'z isize) -> &'z isize {}

fn nested() {
    let y = 3;
    ignore(
        |z| {
            //~^ ERROR E0373
            if false { &y } else { z }
        });
}

fn main() {}
