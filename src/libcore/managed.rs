/*!

Module for wrapping freezable data structures in managed boxes.
Normally freezable data structures require an unaliased reference,
such as `T` or `~T`, so that the compiler can track when they are
being mutated.  The `rw<T>` type converts these static checks into
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
    mut value: T;
    mut mode: Mode;
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
