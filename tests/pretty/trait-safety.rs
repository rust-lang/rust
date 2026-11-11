//@ pp-exact

unsafe trait UnsafeTrait {
    fn foo(&self);
}

unsafe impl UnsafeTrait for isize {
    fn foo(&self) {}
}

pub unsafe trait PubUnsafeTrait {
    fn foo(&self);
}

fn main() {}
