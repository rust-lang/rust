class dog {
    let mut food: uint;

    new() {
        self.food = 0u;
    }

    fn chase_cat() {
        for uint::range(0u, 10u) |_i| {
            let p: &static/mut uint = &mut self.food; //~ ERROR illegal borrow
            *p = 3u;
        }
    }
}

fn main() {
}
