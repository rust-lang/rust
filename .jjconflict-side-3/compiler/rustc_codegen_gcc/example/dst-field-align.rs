// run-pass
#![allow(dead_code)]
struct Foo<T: ?Sized> {
    a: u16,
    b: T
}

trait Bar {
    fn get(&self) -> usize;
}

impl Bar for usize {
    fn get(&self) -> usize { *self }
}

struct Baz<T: ?Sized> {
    a: T
}

struct HasDrop<T: ?Sized> {
    ptr: Box<usize>,
    data: T
}

fn main() {
    // Test that zero-offset works properly
    let b : Baz<usize> = Baz { a: 7 };
    assert_eq!(b.a.get(), 7);
    let b : &Baz<dyn Bar> = &b;
    assert_eq!(b.a.get(), 7);

    // Test that the field is aligned properly
    let f : Foo<usize> = Foo { a: 0, b: 11 };
    assert_eq!(f.b.get(), 11);
    let ptr1 : *const u8 = &f.b as *const _ as *const u8;

    let f : &Foo<dyn Bar> = &f;
    let ptr2 : *const u8 = &f.b as *const _ as *const u8;
    assert_eq!(f.b.get(), 11);

    // The pointers should be the same
    assert_eq!(ptr1, ptr2);

    // Test that nested DSTs work properly
    let f : Foo<Foo<usize>> = Foo { a: 0, b: Foo { a: 1, b: 17 }};
    assert_eq!(f.b.b.get(), 17);
    let f : &Foo<Foo<dyn Bar>> = &f;
    assert_eq!(f.b.b.get(), 17);

    // Test that get the pointer via destructuring works

    let f : Foo<usize> = Foo { a: 0, b: 11 };
    let f : &Foo<dyn Bar> = &f;
    let &Foo { a: _, b: ref bar } = f;
    assert_eq!(bar.get(), 11);

    // Make sure that drop flags don't screw things up

    let d : HasDrop<Baz<[i32; 4]>> = HasDrop {
        ptr: Box::new(0),
        data: Baz { a: [1,2,3,4] }
    };
    assert_eq!([1,2,3,4], d.data.a);

    let d : &HasDrop<Baz<[i32]>> = &d;
    assert_eq!(&[1,2,3,4], &d.data.a);
}
