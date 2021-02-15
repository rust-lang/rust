// run-pass
// pretty-expanded FIXME #23616

pub trait OpInt { fn call(&mut self, _: isize, _: isize) -> isize; }

impl<F> OpInt for F where F: FnMut(isize, isize) -> isize {
    fn call(&mut self, a:isize, b:isize) -> isize {
        (*self)(a, b)
    }
}

fn squarei<'a>(x: isize, op: &'a mut dyn OpInt) -> isize { op.call(x, x) }

fn muli(x:isize, y:isize) -> isize { x * y }

pub fn main() {
    let mut f = |x, y| muli(x, y);
    {
        let g = &mut f;
        let h = g as &mut dyn OpInt;
        squarei(3, h);
    }
}
