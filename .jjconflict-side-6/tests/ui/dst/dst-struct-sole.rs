//@ run-pass
// As dst-struct.rs, but the unsized field is the only field in the struct.


struct Fat<T: ?Sized> {
    ptr: T
}

// x is a fat pointer
fn foo(x: &Fat<[isize]>) {
    let y = &x.ptr;
    assert_eq!(x.ptr.len(), 3);
    assert_eq!(y[0], 1);
    assert_eq!(x.ptr[1], 2);
}

fn foo2<T:ToBar>(x: &Fat<[T]>) {
    let y = &x.ptr;
    let bar = Bar;
    assert_eq!(x.ptr.len(), 3);
    assert_eq!(y[0].to_bar(), bar);
    assert_eq!(x.ptr[1].to_bar(), bar);
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
    let f1 = Fat { ptr: [1, 2, 3] };
    foo(&f1);
    let f2 = &f1;
    foo(f2);
    let f3: &Fat<[isize]> = f2;
    foo(f3);
    let f4: &Fat<[isize]> = &f1;
    foo(f4);
    let f5: &Fat<[isize]> = &Fat { ptr: [1, 2, 3] };
    foo(f5);

    // With a vec of Bars.
    let bar = Bar;
    let f1 = Fat { ptr: [bar, bar, bar] };
    foo2(&f1);
    let f2 = &f1;
    foo2(f2);
    let f3: &Fat<[Bar]> = f2;
    foo2(f3);
    let f4: &Fat<[Bar]> = &f1;
    foo2(f4);
    let f5: &Fat<[Bar]> = &Fat { ptr: [bar, bar, bar] };
    foo2(f5);

    // Assignment.
    let f5: &mut Fat<[isize]> = &mut Fat { ptr: [1, 2, 3] };
    f5.ptr[1] = 34;
    assert_eq!(f5.ptr[0], 1);
    assert_eq!(f5.ptr[1], 34);
    assert_eq!(f5.ptr[2], 3);

    // Zero size vec.
    let f5: &Fat<[isize]> = &Fat { ptr: [] };
    assert!(f5.ptr.is_empty());
    let f5: &Fat<[Bar]> = &Fat { ptr: [] };
    assert!(f5.ptr.is_empty());
}
