// ignore-tidy-linelength
// aux-build:unstable_generic_param.rs

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
    // let _ = S;

    // let _ = Struct1 { field: 1 }; //~ ERROR use of unstable library feature 'unstable_default'
    // let _: Struct1 = Struct1 { field: 1 }; //~ ERROR use of unstable library feature 'unstable_default'
    // let _: Struct1<isize> = Struct1 { field: 1 }; //~ ERROR use of unstable library feature 'unstable_default'

    // let _ = STRUCT1; // ok
    // let _: Struct1 = STRUCT1; // ok
    // let _: Struct1<usize> = STRUCT1; //~ ERROR use of unstable library feature 'unstable_default'
    // let _: Struct1<usize> = STRUCT1; //~ ERROR use of unstable library feature 'unstable_default'
    // let _ = STRUCT1.field; // ok
    // let _: usize = STRUCT1.field; //~ ERROR use of unstable library feature 'unstable_default'
    // let _ = STRUCT1.field + 1; //~ ERROR use of unstable library feature 'unstable_default'
    // let _ = STRUCT1.field + 1usize; //~ ERROR use of unstable library feature 'unstable_default'

    // let _ = Struct2 { field: 1 }; // ok
    // let _: Struct2 = Struct2 { field: 1 }; // ok
    // let _: Struct2<usize> = Struct2 { field: 1 }; // ok

    // let _ = STRUCT2;
    // let _: Struct2 = STRUCT2; // ok
    // let _: Struct2<usize> = STRUCT2; // ok
    // let _: Struct2<usize> = STRUCT2; // ok
    // let _ = STRUCT2.field; // ok
    // let _: usize = STRUCT2.field; // ok
    // let _ = STRUCT2.field + 1; // ok
    // let _ = STRUCT2.field + 1usize; // ok
}
