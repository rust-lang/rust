// Test to make sure that explicit self params work inside closures

struct Box {
    x: uint
}

impl Box {
    fn set_many(&mut self, xs: &[uint]) {
        for xs.each |x| { self.x = *x; }
    }
    fn set_many2(@mut self, xs: &[uint]) {
        for xs.each |x| { self.x = *x; }
    }
    fn set_many3(~mut self, xs: &[uint]) {
        for xs.each |x| { self.x = *x; }
    }
}

fn main() {}
