use std::usize;

struct Dog {
    food: usize,
}

impl Dog {
    pub fn chase_cat(&mut self) {
        let _f = || {
            let p: &'static mut usize = &mut self.food; //~ ERROR cannot infer
            *p = 3;
        };
    }
}

fn main() {
}
