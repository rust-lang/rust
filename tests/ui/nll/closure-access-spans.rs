// check that accesses due to a closure capture give a special note

fn closure_imm_capture_conflict(mut x: i32) {
    let r = &mut x;
    || x; //~ ERROR
    r.use_mut();
}

fn closure_mut_capture_conflict(mut x: i32) {
    let r = &mut x;
    || x = 2; //~ ERROR
    r.use_mut();
}

fn closure_unique_capture_conflict(mut x: &mut i32) {
    let r = &mut x;
    || *x = 2; //~ ERROR
    r.use_mut();
}

fn closure_copy_capture_conflict(mut x: i32) {
    let r = &mut x;
    move || x; //~ ERROR
    r.use_ref();
}

fn closure_move_capture_conflict(mut x: String) {
    let r = &x;
    || x; //~ ERROR
    r.use_ref();
}

fn closure_imm_capture_moved(mut x: String) {
    let r = x;
    || x.len(); //~ ERROR
}

fn closure_mut_capture_moved(mut x: String) {
    let r = x;
    || x = String::new(); //~ ERROR
}

fn closure_unique_capture_moved(x: &mut String) {
    let r = x;
    || *x = String::new(); //~ ERROR
}

fn closure_move_capture_moved(x: &mut String) {
    let r = x;
    || x; //~ ERROR
}

fn main() {}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
