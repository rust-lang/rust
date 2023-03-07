// run-fail
// error-pattern:drop 1
// error-pattern:drop 2
// ignore-emscripten no processes

/// Structure which will not allow to be dropped twice.
struct Droppable<'a>(&'a mut bool, u32);
impl<'a> Drop for Droppable<'a> {
    fn drop(&mut self) {
        if *self.0 {
            eprintln!("{} dropped twice", self.1);
            ::std::process::exit(1);
        }
        eprintln!("drop {}", self.1);
        *self.0 = true;
    }
}

fn mir() {
    let (mut xv, mut yv) = (false, false);
    let x = Droppable(&mut xv, 1);
    let y = Droppable(&mut yv, 2);
    let mut z = x;
    let k = y;
    z = k;
}

fn main() {
    mir();
    panic!();
}
