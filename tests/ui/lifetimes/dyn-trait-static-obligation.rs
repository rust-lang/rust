use std::cell::*;

#[derive(Default)]
struct Test {
    pub foo: u32,
}

trait FooSetter {
    fn set_foo(&mut self, value: u32);
}

impl FooSetter for Test {
    fn set_foo(&mut self, value: u32) {
        self.foo = value;
    }
}

trait BaseSetter{
    fn set(&mut self, value: u32);
}
impl BaseSetter for dyn FooSetter {
    fn set(&mut self, value: u32){
        self.set_foo(value);
    }
}

struct TestHolder<'a> {
    pub holder: Option<RefCell<&'a mut dyn FooSetter>>,
}

impl <'a>TestHolder<'a>{
    pub fn test_foo(&self){
       self.holder.as_ref().unwrap().borrow_mut().set(20);
       //~^ ERROR borrowed data escapes outside of method
    }
}

fn main() {
    let mut test = Test::default();
    test.foo = 10;
    {
        let holder = TestHolder { holder: Some(RefCell::from(&mut test))};

        holder.test_foo();
    }
    test.foo = 30;
}
