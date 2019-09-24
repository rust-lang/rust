// run-pass
// Check that we successfully handle methods where the `self` type has
// an early-bound lifetime. Issue #18208.

// pretty-expanded FIXME #23616

#![allow(dead_code)]

use std::marker;

struct Cursor<'a> {
    m: marker::PhantomData<&'a ()>
}

trait CursorNavigator {
    fn init_cursor<'a, 'b:'a>(&'a self, cursor: &mut Cursor<'b>) -> bool;
}

struct SimpleNavigator;

impl CursorNavigator for SimpleNavigator {
    fn init_cursor<'a, 'b: 'a>(&'a self, _cursor: &mut Cursor<'b>) -> bool {
        false
    }
}

fn main() {
    let mut c = Cursor { m: marker::PhantomData };
    let n = SimpleNavigator;
    n.init_cursor(&mut c);
}
