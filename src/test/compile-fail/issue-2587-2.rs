// xfail-fast
// xfail-test

// XFAIL'd due to problems with error messages on demoded Add.

#[legacy_modes];

fn foo<T: Copy>(+_t: T) { fail; }

fn bar<T>(+_t: T) { fail; }

struct S {
    x: int,
    drop {}
}

fn S(x: int) -> S { S { x: x } }

#[cfg(stage0)]
impl S: Add<S, S> {
    pure fn add(rhs: S) -> S {
        S { x: self.x + rhs.x }
    }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl S : Add<S, S> {
    pure fn add(rhs: &S) -> S {
        S { x: self.x + (*rhs).x }
    }
}

fn main() {
   let v = S(5);
   let _y = v + (move v); //~ ERROR: copying a noncopyable value
}
