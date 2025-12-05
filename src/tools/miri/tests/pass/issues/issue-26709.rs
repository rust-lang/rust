struct Wrapper<'a, T: ?Sized>(&'a mut i32, T);

impl<'a, T: ?Sized> Drop for Wrapper<'a, T> {
    fn drop(&mut self) {
        *self.0 = 432;
    }
}

fn main() {
    let mut x = 0;
    {
        let wrapper = Box::new(Wrapper(&mut x, 123));
        let _val: Box<Wrapper<dyn Send>> = wrapper;
    }
    assert_eq!(432, x)
}
