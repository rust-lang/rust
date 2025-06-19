#![feature(generic_const_exprs)]
#![allow(unused_variables, dead_code, incomplete_features)]

pub struct S;

#[derive(Default)]
pub struct Foo {
    pub bar: S = S, //~ ERROR default values on fields are experimental
    pub baz: i32 = 42 + 3, //~ ERROR default values on fields are experimental
}

#[derive(Default)]
pub enum Bar {
    #[default]
    Foo { //~ ERROR the `#[default]` attribute may only be used on unit enum variants
        bar: S = S, //~ ERROR default values on fields are experimental
        baz: i32 = 42 + 3, //~ ERROR default values on fields are experimental
    }
}

#[derive(Default)]
pub struct Qux<A, const C: i32> {
    bar: S = Qux::<A, C>::S, //~ ERROR default values on fields are experimental
    baz: i32 = foo(), //~ ERROR default values on fields are experimental
    bat: i32 = <Qux<A, C> as T>::K, //~ ERROR default values on fields are experimental
    bay: i32 = C, //~ ERROR default values on fields are experimental
    bak: Vec<A> = Vec::new(), //~ ERROR default values on fields are experimental
}

impl<A, const C: i32> Qux<A, C> {
    const S: S = S;
}

trait T {
    const K: i32;
}

impl<A, const C: i32> T for Qux<A, C> {
    const K: i32 = 2;
}

const fn foo() -> i32 {
    42
}

#[derive(Default)]
pub struct Opt {
    mandatory: Option<()>,
    optional: () = (), //~ ERROR default values on fields are experimental
}

#[derive(Default)]
pub enum OptEnum {
    #[default]
    Variant { //~ ERROR the `#[default]` attribute may only be used on unit enum variants
        mandatory: Option<()>,
        optional: () = (), //~ ERROR default values on fields are experimental
    }
}

// Default field values may not be used on `union`s (at least, this is not described in the accepted
// RFC, and it's not currently clear how to extend the design to do so). We emit a feature gate
// error when the feature is not enabled, but syntactically reject default field values when used
// with unions when the feature is enabled. This can be adjusted if there's an acceptable design
// extension, or just unconditionally reject always.
union U {
    x: i32 = 0,   //~ ERROR default values on fields are experimental
    y: f32 = 0.0, //~ ERROR default values on fields are experimental
}

fn main () {
    let x = Foo { .. }; //~ ERROR base expression required after `..`
    let y = Foo::default();
    let z = Foo { baz: 1, .. }; //~ ERROR base expression required after `..`

    assert_eq!(45, x.baz);
    assert_eq!(45, y.baz);
    assert_eq!(1, z.baz);

    let x = Bar::Foo { .. }; //~ ERROR base expression required after `..`
    let y = Bar::default();
    let z = Bar::Foo { baz: 1, .. }; //~ ERROR base expression required after `..`

    assert!(matches!(Bar::Foo { bar: S, baz: 45 }, x));
    assert!(matches!(Bar::Foo { bar: S, baz: 45 }, y));
    assert!(matches!(Bar::Foo { bar: S, baz: 1 }, z));

    let x = Qux::<i32, 4> { .. }; //~ ERROR base expression required after `..`
    assert!(matches!(Qux::<i32, 4> { bar: S, baz: 42, bat: 2, bay: 4, .. }, x));
    //~^ ERROR base expression required after `..`
    assert!(x.bak.is_empty());
    let y = Opt { mandatory: None, .. };
    //~^ ERROR base expression required after `..`
    assert!(matches!(Opt::default(), y));
    let z = Opt::default();
    assert!(matches!(Opt { mandatory: None, .. }, z));
    //~^ ERROR base expression required after `..`
    assert!(matches!(Opt { .. }, z));
    //~^ ERROR base expression required after `..`
    assert!(matches!(Opt { optional: (), .. }, z));
    //~^ ERROR base expression required after `..`
    assert!(matches!(Opt { optional: (), mandatory: None, .. }, z));
    //~^ ERROR base expression required after `..`
    let y = OptEnum::Variant { mandatory: None, .. };
    //~^ ERROR base expression required after `..`
    assert!(matches!(OptEnum::default(), y));
    let z = OptEnum::default();
    assert!(matches!(OptEnum::Variant { mandatory: None, .. }, z));
    //~^ ERROR base expression required after `..`
    assert!(matches!(OptEnum::Variant { .. }, z));
    //~^ ERROR base expression required after `..`
    assert!(matches!(OptEnum::Variant { optional: (), .. }, z));
    //~^ ERROR base expression required after `..`
    assert!(matches!(OptEnum::Variant { optional: (), mandatory: None, .. }, z));
    //~^ ERROR base expression required after `..`
}
