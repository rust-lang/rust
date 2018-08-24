// error-pattern:unwind happens
// error-pattern:drop 3
// error-pattern:drop 2
// error-pattern:drop 1
// ignore-cloudabi no std::process

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

fn may_panic<'a>() -> Droppable<'a> {
    panic!("unwind happens");
}

fn mir<'a>(d: Droppable<'a>) {
    let (mut a, mut b) = (false, false);
    let y = Droppable(&mut a, 2);
    let x = [Droppable(&mut b, 1), y, d, may_panic()];
}

fn main() {
    let mut c = false;
    mir(Droppable(&mut c, 3));
}
