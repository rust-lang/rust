// ignore-tidy-linelength
// aux-build:unstable_generic_param.rs
#![feature(unstable_default6)]

extern crate unstable_generic_param;

use unstable_generic_param::*;

struct R;

impl Trait1 for S {
    fn foo() -> () { () } // ok
}

struct S;

impl Trait1<usize> for S { //~ ERROR use of unstable library feature 'unstable_default'
    fn foo() -> usize { 0 }
}

impl Trait1<isize> for S { //~ ERROR use of unstable library feature 'unstable_default'
    fn foo() -> isize { 0 }
}

impl Trait2<usize> for S { //~ ERROR use of unstable library feature 'unstable_default'
    fn foo() -> usize { 0 }
}

impl Trait3<usize> for S {
    fn foo() -> usize { 0 } // ok
}

fn main() {
    let _ = S;

    let _: Struct1<isize> = Struct1 { field: 1 }; //~ ERROR use of unstable library feature 'unstable_default'

    let _ = STRUCT1; // ok
    let _: Struct1 = STRUCT1; // ok
    let _: Struct1<usize> = STRUCT1; //~ ERROR use of unstable library feature 'unstable_default'
    let _: Struct1<isize> = Struct1 { field: 0 }; //~ ERROR use of unstable library feature 'unstable_default'

    // Instability is not enforced for generic type parameters used in public fields.
    // Note how the unstable type default `usize` leaks,
    // and can be used without the 'unstable_default' feature.
    let _ = STRUCT1.field;
    let _ = Struct1 { field: 1 };
    let _ = Struct1 { field: () };
    let _ = Struct1 { field: 1isize };
    let _: Struct1 = Struct1 { field: 1 };
    let _: usize = STRUCT1.field;
    let _ = STRUCT1.field + 1;
    let _ = STRUCT1.field + 1usize;

    let _ = Struct2 { field: 1 }; // ok
    let _: Struct2 = Struct2 { field: 1 }; // ok
    let _: Struct2<usize> = Struct2 { field: 1 }; // ok

    let _ = STRUCT2;
    let _: Struct2 = STRUCT2; // ok
    let _: Struct2<usize> = STRUCT2; // ok
    let _: Struct2<isize> = Struct2 { field: 0 }; // ok
    let _ = STRUCT2.field; // ok
    let _: usize = STRUCT2.field; // ok
    let _ = STRUCT2.field + 1; // ok
    let _ = STRUCT2.field + 1usize; // ok

    let _ = STRUCT3;
    let _: Struct3 = STRUCT3; // ok
    let _: Struct3<isize, usize> = STRUCT3; //~ ERROR use of unstable library feature 'unstable_default'
    let _: Struct3<isize> = STRUCT3; // ok
    let _: Struct3<isize, isize> = Struct3 { field1: 0, field2: 0 }; //~ ERROR use of unstable library feature 'unstable_default'
    let _: Struct3<usize, usize> = Struct3 { field1: 0, field2: 0 }; //~ ERROR use of unstable library feature 'unstable_default'
    let _ = STRUCT3.field1; // ok
    let _: isize = STRUCT3.field1; // ok
    let _ = STRUCT3.field1 + 1; // ok
    // Note the aforementioned leak.
    let _: usize = STRUCT3.field2; // ok
    let _: Struct3<usize> = Struct3 { field1: 0, field2: 0 }; // ok
    let _ = STRUCT3.field2 + 1; // ok
    let _ = STRUCT3.field2 + 1usize; // ok

    let _ = STRUCT4;
    let _: Struct4<isize> = Struct4 { field: 1 };
    //~^ use of deprecated item 'unstable_generic_param::Struct4': test [deprecated]
    //~^^ use of deprecated item 'unstable_generic_param::Struct4': test [deprecated]
    //~^^^ use of deprecated item 'unstable_generic_param::Struct4::field': test [deprecated]
    let _ = STRUCT4;
    let _: Struct4 = STRUCT4; //~ use of deprecated item 'unstable_generic_param::Struct4': test [deprecated]
    let _: Struct4<usize> = STRUCT4; //~ use of deprecated item 'unstable_generic_param::Struct4': test [deprecated]
    let _: Struct4<isize> = Struct4 { field: 0 };
    //~^ use of deprecated item 'unstable_generic_param::Struct4': test [deprecated]
    //~^^ use of deprecated item 'unstable_generic_param::Struct4': test [deprecated]
    //~^^^ use of deprecated item 'unstable_generic_param::Struct4::field': test [deprecated]

    let _ = STRUCT5;
    let _: Struct5<isize> = Struct5 { field: 1 }; //~ ERROR use of unstable library feature 'unstable_default'
    //~^ use of deprecated item 'unstable_generic_param::Struct5': test [deprecated]
    //~^^ use of deprecated item 'unstable_generic_param::Struct5': test [deprecated]
    //~^^^ use of deprecated item 'unstable_generic_param::Struct5::field': test [deprecated]
    let _ = STRUCT5;
    let _: Struct5 = STRUCT5; //~ use of deprecated item 'unstable_generic_param::Struct5': test [deprecated]
    let _: Struct5<usize> = STRUCT5; //~ ERROR use of unstable library feature 'unstable_default'
    //~^ use of deprecated item 'unstable_generic_param::Struct5': test [deprecated]
    let _: Struct5<isize> = Struct5 { field: 0 }; //~ ERROR use of unstable library feature 'unstable_default'
    //~^ use of deprecated item 'unstable_generic_param::Struct5': test [deprecated]
    //~^^ use of deprecated item 'unstable_generic_param::Struct5': test [deprecated]
    //~^^^ use of deprecated item 'unstable_generic_param::Struct5::field': test [deprecated]

    let _: Struct6<isize> = Struct6 { field: 1 }; // ok
    let _: Struct6<isize> = Struct6 { field: 0 }; // ok

    let _: Box1<isize, System> = Box1::new(1); //~ ERROR use of unstable library feature 'box_alloc_param'
    let _: Box1<isize> = Box1::new(1); // ok

    let _: Box2<isize, System> = Box2::new(1); // ok
    let _: Box2<isize> = Box2::new(1); // ok

    let _: Box3<isize> = Box3::new(1); // ok
}
