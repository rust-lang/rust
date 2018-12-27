// Note: the borrowck analysis is currently flow-insensitive.
// Therefore, some of these errors are marked as spurious and could be
// corrected by a simple change to the analysis.  The others are
// either genuine or would require more advanced changes.  The latter
// cases are noted.

#![feature(box_syntax)]

fn borrow(_v: &isize) {}
fn borrow_mut(_v: &mut isize) {}
fn cond() -> bool { panic!() }
fn for_func<F>(_f: F) where F: FnOnce() -> bool { panic!() }
fn produce<T>() -> T { panic!(); }

fn inc(v: &mut Box<isize>) {
    *v = box (**v + 1);
}

fn pre_freeze() {
    // In this instance, the freeze starts before the mut borrow.

    let mut v: Box<_> = box 3;
    let _w = &v;
    borrow_mut(&mut *v); //~ ERROR cannot borrow
    _w.use_ref();
}

fn post_freeze() {
    // In this instance, the const alias starts after the borrow.

    let mut v: Box<_> = box 3;
    borrow_mut(&mut *v);
    let _w = &v;
}

fn main() {}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
