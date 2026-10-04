//@ run-fail
//@ error-pattern:drop 1
//@ needs-subprocess

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

fn mir<'a>(d: Droppable<'a>) {
    loop {
        let x = d;
        break;
    }
}

fn main() {
    let mut xv = false;
    mir(Droppable(&mut xv, 1));
    panic!();
}
