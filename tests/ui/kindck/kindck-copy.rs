// Test which of the builtin types are considered POD.

use std::rc::Rc;

fn assert_copy<T:Copy>() { }

trait Dummy { }

#[derive(Copy, Clone)]
struct MyStruct {
    x: isize,
    y: isize,
}

struct MyNoncopyStruct {
    x: Box<char>,
}

fn test<'a,T,U:Copy>(_: &'a isize) {
    // lifetime pointers are ok...
    assert_copy::<&'static isize>();
    assert_copy::<&'a isize>();
    assert_copy::<&'a str>();
    assert_copy::<&'a [isize]>();

    // ...unless they are mutable
    assert_copy::<&'static mut isize>(); //~ ERROR : Copy` is not satisfied
    assert_copy::<&'a mut isize>();  //~ ERROR : Copy` is not satisfied

    // boxes are not ok
    assert_copy::<Box<isize>>();   //~ ERROR : Copy` is not satisfied
    assert_copy::<String>();   //~ ERROR : Copy` is not satisfied
    assert_copy::<Vec<isize> >(); //~ ERROR : Copy` is not satisfied
    assert_copy::<Box<&'a mut isize>>(); //~ ERROR : Copy` is not satisfied

    // borrowed object types are generally ok
    assert_copy::<&'a dyn Dummy>();
    assert_copy::<&'a (dyn Dummy + Send)>();
    assert_copy::<&'static (dyn Dummy + Send)>();

    // owned object types are not ok
    assert_copy::<Box<dyn Dummy>>(); //~ ERROR : Copy` is not satisfied
    assert_copy::<Box<dyn Dummy + Send>>(); //~ ERROR : Copy` is not satisfied

    // mutable object types are not ok
    assert_copy::<&'a mut (dyn Dummy + Send)>();  //~ ERROR : Copy` is not satisfied

    // raw ptrs are ok
    assert_copy::<*const isize>();
    assert_copy::<*const &'a mut isize>();

    // regular old ints and such are ok
    assert_copy::<isize>();
    assert_copy::<bool>();
    assert_copy::<()>();

    // tuples are ok
    assert_copy::<(isize,isize)>();

    // structs of POD are ok
    assert_copy::<MyStruct>();

    // structs containing non-POD are not ok
    assert_copy::<MyNoncopyStruct>(); //~ ERROR : Copy` is not satisfied

    // ref counted types are not ok
    assert_copy::<Rc<isize>>();   //~ ERROR : Copy` is not satisfied
}

pub fn main() {
}
