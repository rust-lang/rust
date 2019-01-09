// run-pass
// Tests that the scope of the pointer returned from `get()` is
// limited to the deref operation itself, and does not infect the
// block as a whole.


struct Box {
    x: usize
}

impl Box {
    fn get(&self) -> &usize {
        &self.x
    }
    fn set(&mut self, x: usize) {
        self.x = x;
    }
}

fn fun1() {
    // in the past, borrow checker behaved differently when
    // init and decl of `v` were distinct
    let v;
    let mut a_box = Box {x: 0};
    a_box.set(22);
    v = *a_box.get();
    a_box.set(v+1);
    assert_eq!(23, *a_box.get());
}

fn fun2() {
    let mut a_box = Box {x: 0};
    a_box.set(22);
    let v = *a_box.get();
    a_box.set(v+1);
    assert_eq!(23, *a_box.get());
}

pub fn main() {
    fun1();
    fun2();
}
