#[derive(Clone, Copy)]
struct S {
    a: u8,
    b: u16,
}

#[derive(Clone, Copy)]
union U {
    s: S,
    c: u32,
}

fn main() {
    unsafe {
        {
            let mut u = U { s: S { a: 0, b: 1 } };
            let ra = &mut u.s.a;
            let b = u.s.b; // OK
            ra.use_mut();
        }
        {
            let mut u = U { s: S { a: 0, b: 1 } };
            let ra = &mut u.s.a;
            let b = u.c; //~ ERROR cannot use `u.c` because it was mutably borrowed
            ra.use_mut();
        }
    }
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
