#![feature(rustc_attrs)]
macro_rules! width(
    ($this:expr) => {
        $this.width.unwrap()
        //~^ ERROR cannot use `self.width` because it was mutably borrowed
    }
);

struct HasInfo {
    width: Option<usize>
}

impl HasInfo {
    fn get_size(&mut self, n: usize) -> usize {
        n
    }

    fn get_other(&mut self) -> usize {
        let r = &mut *self;
        r.get_size(width!(self))
    }
    // Above is like `self.get_size(width!(self))`, but it
    // deliberately avoids NLL's two phase borrow feature.
}

fn main() { }
