//@ run-fail
//@ error-pattern:diverging_fn called
//@ error-pattern:0 dropped
//@ needs-unwind this test checks that a destructor is called after panicking

struct Droppable(u8);
impl Drop for Droppable {
    fn drop(&mut self) {
        eprintln!("{} dropped", self.0);
    }
}

fn diverging_fn() -> ! {
    panic!("diverging_fn called")
}

fn mir(d: Droppable) {
    diverging_fn();
}

fn main() {
    let d = Droppable(0);
    mir(d);
}
