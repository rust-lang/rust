//@ check-pass

fn method<'a>(_i: &'a i32) {
    fn inner<'a>(_j: &'a f32) {}
}

fn main() {}
