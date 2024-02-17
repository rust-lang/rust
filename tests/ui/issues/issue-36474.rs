//@ run-pass
fn main() {
    remove_axis(&3, 0);
}

trait Dimension {
    fn slice(&self) -> &[usize];
}

impl Dimension for () {
    fn slice(&self) -> &[usize] { &[] }
}

impl Dimension for usize {
    fn slice(&self) -> &[usize] {
        unsafe {
            ::std::slice::from_raw_parts(self, 1)
        }
    }
}

fn remove_axis(value: &usize, axis: usize) -> () {
    let tup = ();
    let mut it = tup.slice().iter();
    for (i, _) in value.slice().iter().enumerate() {
        if i == axis {
            continue;
        }
        it.next();
    }
}
