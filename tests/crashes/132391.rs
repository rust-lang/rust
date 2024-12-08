//@ known-bug: #123291

trait MyTrait {
    #[repr(align)]
    fn myfun();
}

pub fn main() {}
