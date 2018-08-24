// pretty-expanded FIXME #23616

mod foo {
    pub enum t { t1, }
}

pub fn main() { let _v = foo::t::t1; }
