use std::ops::Add;

#[derive(Copy, Clone)]
struct Point {
    x: isize,
    y: isize,
}

impl Add<isize> for Point {
    type Output = isize;

    fn add(self, z: isize) -> isize {
        self.x + self.y + z
    }
}

impl Point {
    pub fn times(&self, z: isize) -> isize {
        self.x * self.y * z
    }
}

fn a() {
    let mut p = Point {x: 3, y: 4};

    // ok (we can loan out rcvr)
    p + 3;
    p.times(3);
}

fn b() {
    let mut p = Point {x: 3, y: 4};

    // Here I create an outstanding loan and check that we get conflicts:

    let q = &mut p;

    p + 3;  //~ ERROR cannot use `p`
    p.times(3); //~ ERROR cannot borrow `p`

    *q + 3; // OK to use the new alias `q`
    q.x += 1; // and OK to mutate it
}

fn main() {
}
