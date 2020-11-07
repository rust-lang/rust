#![warn(clippy::let_underscore_drop)]

struct Droppable;

impl Drop for Droppable {
    fn drop(&mut self) {}
}

fn main() {
    let unit = ();
    let boxed = Box::new(());
    let droppable = Droppable;
    let optional = Some(Droppable);

    let _ = ();
    let _ = Box::new(());
    let _ = Droppable;
    let _ = Some(Droppable);
}
