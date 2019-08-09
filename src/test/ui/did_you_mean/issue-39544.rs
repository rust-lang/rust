pub enum X {
    Y
}

pub struct Z {
    x: X
}

fn main() {
    let z = Z { x: X::Y };
    let _ = &mut z.x; //~ ERROR cannot borrow
}

impl Z {
    fn foo<'z>(&'z self) {
        let _ = &mut self.x; //~ ERROR cannot borrow
    }

    fn foo1(&self, other: &Z) {
        let _ = &mut self.x; //~ ERROR cannot borrow
        let _ = &mut other.x; //~ ERROR cannot borrow
    }

    fn foo2<'a>(&'a self, other: &Z) {
        let _ = &mut self.x; //~ ERROR cannot borrow
        let _ = &mut other.x; //~ ERROR cannot borrow
    }

    fn foo3<'a>(self: &'a Self, other: &Z) {
        let _ = &mut self.x; //~ ERROR cannot borrow
        let _ = &mut other.x; //~ ERROR cannot borrow
    }

    fn foo4(other: &Z) {
        let _ = &mut other.x; //~ ERROR cannot borrow
    }

}

pub fn with_arg(z: Z, w: &Z) {
    let _ = &mut z.x; //~ ERROR cannot borrow
    let _ = &mut w.x; //~ ERROR cannot borrow
}

pub fn with_tuple() {
    let mut y = 0;
    let x = (&y,);
    *x.0 = 1;
    //~^ ERROR cannot assign to `*x.0` which is behind a `&` reference
}
