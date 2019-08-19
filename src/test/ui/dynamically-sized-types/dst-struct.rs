// run-pass
#![feature(box_syntax)]

struct Fat<T: ?Sized> {
    f1: isize,
    f2: &'static str,
    ptr: T
}

// x is a fat pointer
fn foo(x: &Fat<[isize]>) {
    let y = &x.ptr;
    assert_eq!(x.ptr.len(), 3);
    assert_eq!(y[0], 1);
    assert_eq!(x.ptr[1], 2);
    assert_eq!(x.f1, 5);
    assert_eq!(x.f2, "some str");
}

fn foo2<T:ToBar>(x: &Fat<[T]>) {
    let y = &x.ptr;
    let bar = Bar;
    assert_eq!(x.ptr.len(), 3);
    assert_eq!(y[0].to_bar(), bar);
    assert_eq!(x.ptr[1].to_bar(), bar);
    assert_eq!(x.f1, 5);
    assert_eq!(x.f2, "some str");
}

fn foo3(x: &Fat<Fat<[isize]>>) {
    let y = &x.ptr.ptr;
    assert_eq!(x.f1, 5);
    assert_eq!(x.f2, "some str");
    assert_eq!(x.ptr.f1, 8);
    assert_eq!(x.ptr.f2, "deep str");
    assert_eq!(x.ptr.ptr.len(), 3);
    assert_eq!(y[0], 1);
    assert_eq!(x.ptr.ptr[1], 2);
}


#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Bar;

trait ToBar {
    fn to_bar(&self) -> Bar;
}

impl ToBar for Bar {
    fn to_bar(&self) -> Bar {
        *self
    }
}

pub fn main() {
    // With a vec of ints.
    let f1 = Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    foo(&f1);
    let f2 = &f1;
    foo(f2);
    let f3: &Fat<[isize]> = f2;
    foo(f3);
    let f4: &Fat<[isize]> = &f1;
    foo(f4);
    let f5: &Fat<[isize]> = &Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    foo(f5);

    // With a vec of Bars.
    let bar = Bar;
    let f1 = Fat { f1: 5, f2: "some str", ptr: [bar, bar, bar] };
    foo2(&f1);
    let f2 = &f1;
    foo2(f2);
    let f3: &Fat<[Bar]> = f2;
    foo2(f3);
    let f4: &Fat<[Bar]> = &f1;
    foo2(f4);
    let f5: &Fat<[Bar]> = &Fat { f1: 5, f2: "some str", ptr: [bar, bar, bar] };
    foo2(f5);

    // Assignment.
    let f5: &mut Fat<[isize]> = &mut Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    f5.ptr[1] = 34;
    assert_eq!(f5.ptr[0], 1);
    assert_eq!(f5.ptr[1], 34);
    assert_eq!(f5.ptr[2], 3);

    // Zero size vec.
    let f5: &Fat<[isize]> = &Fat { f1: 5, f2: "some str", ptr: [] };
    assert!(f5.ptr.is_empty());
    let f5: &Fat<[Bar]> = &Fat { f1: 5, f2: "some str", ptr: [] };
    assert!(f5.ptr.is_empty());

    // Deeply nested.
    let f1 = Fat { f1: 5, f2: "some str", ptr: Fat { f1: 8, f2: "deep str", ptr: [1, 2, 3]} };
    foo3(&f1);
    let f2 = &f1;
    foo3(f2);
    let f3: &Fat<Fat<[isize]>> = f2;
    foo3(f3);
    let f4: &Fat<Fat<[isize]>> = &f1;
    foo3(f4);
    let f5: &Fat<Fat<[isize]>> =
        &Fat { f1: 5, f2: "some str", ptr: Fat { f1: 8, f2: "deep str", ptr: [1, 2, 3]} };
    foo3(f5);

    // Box.
    let f1 = Box::new([1, 2, 3]);
    assert_eq!((*f1)[1], 2);
    let f2: Box<[isize]> = f1;
    assert_eq!((*f2)[1], 2);

    // Nested Box.
    let f1 : Box<Fat<[isize; 3]>> = box Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    foo(&*f1);
    let f2 : Box<Fat<[isize]>> = f1;
    foo(&*f2);

    let f3 : Box<Fat<[isize]>> =
        Box::<Fat<[_; 3]>>::new(Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] });
    foo(&*f3);
}
