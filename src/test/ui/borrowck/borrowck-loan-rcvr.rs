struct Point { x: isize, y: isize }

trait Methods {
    fn impurem(&self);
    fn blockm<F>(&self, f: F) where F: FnOnce();
}

impl Methods for Point {
    fn impurem(&self) {
    }

    fn blockm<F>(&self, f: F) where F: FnOnce() { f() }
}

fn a() {
    let mut p = Point {x: 3, y: 4};

    // Here: it's ok to call even though receiver is mutable, because we
    // can loan it out.
    p.impurem();

    // But in this case we do not honor the loan:
    p.blockm(|| { //~ ERROR cannot borrow `p` as mutable
        p.x = 10;
    })
}

fn b() {
    let mut p = Point {x: 3, y: 4};

    // Here I create an outstanding loan and check that we get conflicts:

    let l = &mut p;
    p.impurem(); //~ ERROR cannot borrow

    l.x += 1;
}

fn main() {
}
