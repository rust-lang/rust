// run-pass
#![allow(type_alias_bounds)]

#![feature(box_syntax)]
#![feature(unsized_tuple_coercion)]

type Fat<T: ?Sized> = (isize, &'static str, T);

// x is a fat pointer
fn foo(x: &Fat<[isize]>) {
    let y = &x.2;
    assert_eq!(x.2.len(), 3);
    assert_eq!(y[0], 1);
    assert_eq!(x.2[1], 2);
    assert_eq!(x.0, 5);
    assert_eq!(x.1, "some str");
}

fn foo2<T:ToBar>(x: &Fat<[T]>) {
    let y = &x.2;
    let bar = Bar;
    assert_eq!(x.2.len(), 3);
    assert_eq!(y[0].to_bar(), bar);
    assert_eq!(x.2[1].to_bar(), bar);
    assert_eq!(x.0, 5);
    assert_eq!(x.1, "some str");
}

fn foo3(x: &Fat<Fat<[isize]>>) {
    let y = &(x.2).2;
    assert_eq!(x.0, 5);
    assert_eq!(x.1, "some str");
    assert_eq!((x.2).0, 8);
    assert_eq!((x.2).1, "deep str");
    assert_eq!((x.2).2.len(), 3);
    assert_eq!(y[0], 1);
    assert_eq!((x.2).2[1], 2);
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
    let f1 = (5, "some str", [1, 2, 3]);
    foo(&f1);
    let f2 = &f1;
    foo(f2);
    let f3: &Fat<[isize]> = f2;
    foo(f3);
    let f4: &Fat<[isize]> = &f1;
    foo(f4);
    let f5: &Fat<[isize]> = &(5, "some str", [1, 2, 3]);
    foo(f5);

    // With a vec of Bars.
    let bar = Bar;
    let f1 = (5, "some str", [bar, bar, bar]);
    foo2(&f1);
    let f2 = &f1;
    foo2(f2);
    let f3: &Fat<[Bar]> = f2;
    foo2(f3);
    let f4: &Fat<[Bar]> = &f1;
    foo2(f4);
    let f5: &Fat<[Bar]> = &(5, "some str", [bar, bar, bar]);
    foo2(f5);

    // Assignment.
    let f5: &mut Fat<[isize]> = &mut (5, "some str", [1, 2, 3]);
    f5.2[1] = 34;
    assert_eq!(f5.2[0], 1);
    assert_eq!(f5.2[1], 34);
    assert_eq!(f5.2[2], 3);

    // Zero size vec.
    let f5: &Fat<[isize]> = &(5, "some str", []);
    assert!(f5.2.is_empty());
    let f5: &Fat<[Bar]> = &(5, "some str", []);
    assert!(f5.2.is_empty());

    // Deeply nested.
    let f1 = (5, "some str", (8, "deep str", [1, 2, 3]));
    foo3(&f1);
    let f2 = &f1;
    foo3(f2);
    let f3: &Fat<Fat<[isize]>> = f2;
    foo3(f3);
    let f4: &Fat<Fat<[isize]>> = &f1;
    foo3(f4);
    let f5: &Fat<Fat<[isize]>> = &(5, "some str", (8, "deep str", [1, 2, 3]));
    foo3(f5);

    // Box.
    let f1 = Box::new([1, 2, 3]);
    assert_eq!((*f1)[1], 2);
    let f2: Box<[isize]> = f1;
    assert_eq!((*f2)[1], 2);

    // Nested Box.
    let f1 : Box<Fat<[isize; 3]>> = box (5, "some str", [1, 2, 3]);
    foo(&*f1);
    let f2 : Box<Fat<[isize]>> = f1;
    foo(&*f2);

    let f3 : Box<Fat<[isize]>> =
        Box::<Fat<[_; 3]>>::new((5, "some str", [1, 2, 3]));
    foo(&*f3);
}
