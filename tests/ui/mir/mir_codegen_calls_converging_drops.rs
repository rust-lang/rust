//@ run-fail
//@ check-run-results:converging_fn called
//@ check-run-results:0 dropped
//@ check-run-results:exit
//@ ignore-emscripten no processes

struct Droppable(u8);
impl Drop for Droppable {
    fn drop(&mut self) {
        eprintln!("{} dropped", self.0);
    }
}

fn converging_fn() {
    eprintln!("converging_fn called");
}

fn mir(d: Droppable) {
    converging_fn();
}

fn main() {
    let d = Droppable(0);
    mir(d);
    panic!("exit");
}
