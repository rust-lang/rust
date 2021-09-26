// run-pass

#![allow(non_camel_case_types)]
#![allow(dead_code)]
// pretty-expanded FIXME #23616

fn p_foo<T>(_pinned: T) { }
fn s_foo<T>(_shared: T) { }
fn u_foo<T:Send>(_unique: T) { }

struct r {
  i: isize,
}

impl Drop for r {
    fn drop(&mut self) {}
}

fn r(i:isize) -> r {
    r {
        i: i
    }
}

pub fn main() {
    p_foo(r(10));

    p_foo::<Box<_>>(Box::new(r(10)));
    p_foo::<Box<_>>(Box::new(10));
    p_foo(10);

    s_foo::<Box<_>>(Box::new(10));
    s_foo(10);

    u_foo::<Box<_>>(Box::new(10));
    u_foo(10);
}
