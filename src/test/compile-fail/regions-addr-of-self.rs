struct dog {
    let mut cats_chased: uint;

    new() {
        self.cats_chased = 0u;
    }

    fn chase_cat() {
        let p: &static/mut uint = &mut self.cats_chased; //~ ERROR illegal borrow
        *p += 1u;
    }

    fn chase_cat_2() {
        let p: &blk/mut uint = &mut self.cats_chased;
        *p += 1u;
    }
}

fn main() {
    let d = dog();
    d.chase_cat();
    debug!("cats_chased: %u", d.cats_chased);
}

