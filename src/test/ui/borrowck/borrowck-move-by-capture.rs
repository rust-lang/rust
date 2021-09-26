#![feature(unboxed_closures)]

fn to_fn_mut<A,F:FnMut<A>>(f: F) -> F { f }
fn to_fn_once<A,F:FnOnce<A>>(f: F) -> F { f }

pub fn main() {
    let bar: Box<_> = Box::new(3);
    let _g = to_fn_mut(|| {
        let _h = to_fn_once(move || -> isize { *bar }); //~ ERROR cannot move out of
    });
}
