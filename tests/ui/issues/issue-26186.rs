//@ check-pass
use std::sync::Mutex;
use std::cell::RefCell;
use std::rc::Rc;
use std::ops::*;

//eefriedman example
struct S<'a, T:FnMut() + 'static + ?Sized>(&'a mut T);
impl<'a, T:?Sized + FnMut() + 'static> DerefMut for S<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}
impl<'a, T:?Sized + FnMut() + 'static> Deref for S<'a, T> {
    type Target = dyn FnMut() + 'a;
    fn deref(&self) -> &Self::Target { &self.0 }
}

//Ossipal example
struct FunctionIcon {
    get_icon: Mutex<Box<dyn FnMut() -> u32>>,
}

impl FunctionIcon {
    fn get_icon(&self) -> impl '_ + std::ops::DerefMut<Target=Box<dyn FnMut() -> u32>> {
        self.get_icon.lock().unwrap()
    }

    fn load_icon(&self)  {
        let mut get_icon = self.get_icon();
        let _rgba_icon = (*get_icon)();
    }
}

//shepmaster example
struct Foo;

impl Deref for Foo {
    type Target = dyn FnMut() + 'static;
    fn deref(&self) -> &Self::Target {
        unimplemented!()
    }
}

impl DerefMut for Foo {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unimplemented!()
    }
}

fn main() {
    //eefriedman example
    let mut f = ||{};
    let mut s = S(&mut f);
    s();

    //Diggsey/Mark-Simulacrum example
    let a: Rc<RefCell<dyn FnMut()>> = Rc::new(RefCell::new(||{}));
    a.borrow_mut()();

    //shepmaster example
    let mut t = Foo;
    t();
}
