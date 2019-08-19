// run-pass
#![allow(type_alias_bounds)]

#![allow(unused_features)]
#![feature(box_syntax)]
#![feature(unsized_tuple_coercion)]

type Fat<T: ?Sized> = (isize, &'static str, T);

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Bar;

#[derive(Copy, Clone, PartialEq, Eq)]
struct Bar1 {
    f: isize
}

trait ToBar {
    fn to_bar(&self) -> Bar;
    fn to_val(&self) -> isize;
}

impl ToBar for Bar {
    fn to_bar(&self) -> Bar {
        *self
    }
    fn to_val(&self) -> isize {
        0
    }
}
impl ToBar for Bar1 {
    fn to_bar(&self) -> Bar {
        Bar
    }
    fn to_val(&self) -> isize {
        self.f
    }
}

// x is a fat pointer
fn foo(x: &Fat<dyn ToBar>) {
    assert_eq!(x.0, 5);
    assert_eq!(x.1, "some str");
    assert_eq!(x.2.to_bar(), Bar);
    assert_eq!(x.2.to_val(), 42);

    let y = &x.2;
    assert_eq!(y.to_bar(), Bar);
    assert_eq!(y.to_val(), 42);
}

fn bar(x: &dyn ToBar) {
    assert_eq!(x.to_bar(), Bar);
    assert_eq!(x.to_val(), 42);
}

fn baz(x: &Fat<Fat<dyn ToBar>>) {
    assert_eq!(x.0, 5);
    assert_eq!(x.1, "some str");
    assert_eq!((x.2).0, 8);
    assert_eq!((x.2).1, "deep str");
    assert_eq!((x.2).2.to_bar(), Bar);
    assert_eq!((x.2).2.to_val(), 42);

    let y = &(x.2).2;
    assert_eq!(y.to_bar(), Bar);
    assert_eq!(y.to_val(), 42);

}

pub fn main() {
    let f1 = (5, "some str", Bar1 {f :42});
    foo(&f1);
    let f2 = &f1;
    foo(f2);
    let f3: &Fat<dyn ToBar> = f2;
    foo(f3);
    let f4: &Fat<dyn ToBar> = &f1;
    foo(f4);
    let f5: &Fat<dyn ToBar> = &(5, "some str", Bar1 {f :42});
    foo(f5);

    // Zero size object.
    let f6: &Fat<dyn ToBar> = &(5, "some str", Bar);
    assert_eq!(f6.2.to_bar(), Bar);

    // &*
    //
    let f7: Box<dyn ToBar> = Box::new(Bar1 {f :42});
    bar(&*f7);

    // Deep nesting
    let f1 = (5, "some str", (8, "deep str", Bar1 {f :42}));
    baz(&f1);
    let f2 = &f1;
    baz(f2);
    let f3: &Fat<Fat<dyn ToBar>> = f2;
    baz(f3);
    let f4: &Fat<Fat<dyn ToBar>> = &f1;
    baz(f4);
    let f5: &Fat<Fat<dyn ToBar>> = &(5, "some str", (8, "deep str", Bar1 {f :42}));
    baz(f5);
}
