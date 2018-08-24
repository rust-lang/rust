struct Obj {
    member: usize
}

impl Obj {
    pub fn boom() -> bool {
        return 1+1 == 2
    }
    pub fn chirp(&self) {
        self.boom(); //~ ERROR no method named `boom` found for type `&Obj` in the current scope
    }
}

fn main() {
    let o = Obj { member: 0 };
    o.chirp();
    1 + 1;
}
