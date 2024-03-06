#![allow(incomplete_features)]
#![feature(unnamed_fields)]

struct A { //~ ERROR struct with unnamed fields must have `#[repr(C)]` representation
           //~^ NOTE struct `A` defined here
    _: struct { //~ NOTE unnamed field defined here
        a: i32,
    },
    _: struct { //~ NOTE unnamed field defined here
        _: struct {
            b: i32,
        },
    },
}

union B { //~ ERROR union with unnamed fields must have `#[repr(C)]` representation
          //~^ NOTE union `B` defined here
    _: union { //~ NOTE unnamed field defined here
        a: i32,
    },
    _: union { //~ NOTE unnamed field defined here
        _: union {
            b: i32,
        },
    },
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Foo {}

#[derive(Clone, Copy)]
struct Bar {}
//~^ `Bar` defined here
//~| `Bar` defined here
//~| `Bar` defined here
//~| `Bar` defined here

struct C { //~ ERROR struct with unnamed fields must have `#[repr(C)]` representation
           //~^ NOTE struct `C` defined here
    _: Foo, //~ NOTE unnamed field defined here
}

union D { //~ ERROR union with unnamed fields must have `#[repr(C)]` representation
          //~^ NOTE union `D` defined here
    _: Foo, //~ NOTE unnamed field defined here
}

#[repr(C)]
struct E {
    _: Bar, //~ ERROR named type of unnamed field must have `#[repr(C)]` representation
            //~^ NOTE unnamed field defined here
    _: struct {
        _: Bar, //~ ERROR named type of unnamed field must have `#[repr(C)]` representation
                //~^ NOTE unnamed field defined here
    },
}

#[repr(C)]
union F {
    _: Bar, //~ ERROR named type of unnamed field must have `#[repr(C)]` representation
            //~^ NOTE unnamed field defined here
    _: union {
        _: Bar, //~ ERROR named type of unnamed field must have `#[repr(C)]` representation
                //~^ NOTE unnamed field defined here
    },
}

fn main() {}
