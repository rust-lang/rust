// Test that we don't show variables with from for loop desugaring

fn for_loop(s: &[i32]) {
    for &ref mut x in s {}
    //~^ ERROR cannot borrow data in a `&` reference as mutable [E0596]
}

struct D<'a>(&'a ());

impl Drop for D<'_> {
    fn drop(&mut self) {}
}

fn for_loop_dropck(v: Vec<D<'static>>) {
    for ref mut d in v {
        let y = ();
        *d = D(&y); //~ ERROR `y` does not live long enough
    }
}

fn main() {}
