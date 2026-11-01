// Note: the borrowck analysis is currently flow-insensitive.
// Therefore, some of these errors are marked as spurious and could be
// corrected by a simple change to the analysis.  The others are
// either genuine or would require more advanced changes.  The latter
// cases are noted.



fn borrow(_v: &isize) {}
fn borrow_mut(_v: &mut isize) {}
fn cond() -> bool { panic!() }
fn for_func<F>(_f: F) where F: FnOnce() -> bool { panic!() }
fn produce<T>() -> T { panic!(); }

fn inc(v: &mut Box<isize>) {
    *v = Box::new(**v + 1);
}

fn pre_freeze_cond() {
    // In this instance, the freeze is conditional and starts before
    // the mut borrow.

    let u = Box::new(0);
    let mut v: Box<_> = Box::new(3);
    let mut _w = &u;
    if cond() {
        _w = &v;
    }
    borrow_mut(&mut *v); //~ ERROR cannot borrow
    _w.use_ref();
}

fn pre_freeze_else() {
    // In this instance, the freeze and mut borrow are on separate sides
    // of the if.

    let u = Box::new(0);
    let mut v: Box<_> = Box::new(3);
    let mut _w = &u;
    if cond() {
        _w = &v;
    } else {
        borrow_mut(&mut *v);
    }
    _w.use_ref();
}

fn main() {}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
