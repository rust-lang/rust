// error-pattern:panic 1
// error-pattern:drop 2

struct Droppable(u32);
impl Drop for Droppable {
    fn drop(&mut self) {
        if self.0 == 1 {
            panic!("panic 1");
        } else {
            eprintln!("drop {}", self.0);
        }
    }
}

fn mir() {
    let x = Droppable(2);
    let y = Droppable(1);
}

fn main() {
    mir();
}
