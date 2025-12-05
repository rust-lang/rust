struct Thing {
    x: isize
}

impl Thing {
    fn mul(&self, c: &isize) -> Thing {
        Thing {x: self.x * *c}
    }
}

fn main() {
    let u = Thing {x: 2};
    let _v = u.mul(&3); // This is ok
    let w = u * 3; //~ ERROR cannot multiply `Thing` by `{integer}`
}
