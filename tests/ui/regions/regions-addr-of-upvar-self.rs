struct Dog {
    food: usize,
}

impl Dog {
    pub fn chase_cat(&mut self) {
        let _f = || {
            let p: &'static mut usize = &mut self.food;
            //~^ ERROR lifetime may not live long enough
            //~^^ ERROR lifetime may not live long enough
            //~^^^ ERROR E0597
            *p = 3;
        };
    }
}

fn main() {
}
