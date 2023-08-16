// run-fail
//@error-in-other-file:converging_fn called
//@error-in-other-file:0 dropped
//@error-in-other-file:exit
//@ignore-target-emscripten no processes

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
