#![feature(generators)]

enum Test { A(i32), B, }

fn main() { }

fn fun(test: Test) {
    move || {
        if let Test::A(ref _a) = test { //~ ERROR borrow may still be in use when generator yields
            yield ();
            _a.use_ref();
        }
    };
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
