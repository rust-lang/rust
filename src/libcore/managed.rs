/*!

Module for wrapping freezable data structures in managed boxes.
Normally freezable data structures require an unaliased reference,
such as `T` or `~T`, so that the compiler can track when they are
being mutated.  The `managed<T>` type converts these static checks into
dynamic checks: your program will fail if you attempt to perform
mutation when the data structure should be immutable.

*/

#[forbid(non_camel_case_types)];
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

import util::with;
import unsafe::transmute_immut;

export Managed;

enum Mode { ReadOnly, Mutable, Immutable }

struct Data<T> {
    priv mut value: T;
    priv mut mode: Mode;
}

type Managed<T> = @Data<T>;

fn Managed<T>(+t: T) -> Managed<T> {
    @Data {value: t, mode: ReadOnly}
}

impl<T> Data<T> {
    fn borrow_mut<R>(op: &fn(t: &mut T) -> R) -> R {
        match self.mode {
          Immutable => fail fmt!("%? currently immutable",
                                 self.value),
          ReadOnly | Mutable => {}
        }

        do with(&mut self.mode, Mutable) {
            op(&mut self.value)
        }
    }

    fn borrow_const<R>(op: &fn(t: &const T) -> R) -> R {
        op(&const self.value)
    }

    fn borrow_imm<R>(op: &fn(t: &T) -> R) -> R {
        match self.mode {
          Mutable => fail fmt!("%? currently mutable",
                               self.value),
          ReadOnly | Immutable => {}
        }

        do with(&mut self.mode, Immutable) {
            op(unsafe{transmute_immut(&mut self.value)})
        }
    }
}

#[test]
#[should_fail]
fn test_mut_in_imm() {
    let m = Managed(1);
    do m.borrow_imm |_p| {
        do m.borrow_mut |_q| {
            // should not be permitted
        }
    }
}

#[test]
#[should_fail]
fn test_imm_in_mut() {
    let m = Managed(1);
    do m.borrow_mut |_p| {
        do m.borrow_imm |_q| {
            // should not be permitted
        }
    }
}

#[test]
fn test_const_in_mut() {
    let m = Managed(1);
    do m.borrow_mut |p| {
        do m.borrow_const |q| {
            assert *p == *q;
            *p += 1;
            assert *p == *q;
        }
    }
}

#[test]
fn test_mut_in_const() {
    let m = Managed(1);
    do m.borrow_const |p| {
        do m.borrow_mut |q| {
            assert *p == *q;
            *q += 1;
            assert *p == *q;
        }
    }
}

#[test]
fn test_imm_in_const() {
    let m = Managed(1);
    do m.borrow_const |p| {
        do m.borrow_imm |q| {
            assert *p == *q;
        }
    }
}

#[test]
fn test_const_in_imm() {
    let m = Managed(1);
    do m.borrow_imm |p| {
        do m.borrow_const |q| {
            assert *p == *q;
        }
    }
}


#[test]
#[should_fail]
fn test_mut_in_imm_in_const() {
    let m = Managed(1);
    do m.borrow_const |_p| {
        do m.borrow_imm |_q| {
            do m.borrow_mut |_r| {
            }
        }
    }
}
