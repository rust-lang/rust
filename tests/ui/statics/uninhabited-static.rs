#![feature(never_type)]
#![deny(uninhabited_static)]

enum Void {}
extern "C" {
    static VOID: Void; //~ ERROR static of uninhabited type
    //~| WARN: previously accepted
    static NEVER: !; //~ ERROR static of uninhabited type
    //~| WARN: previously accepted
}

static VOID2: Void = unsafe { std::mem::transmute(()) }; //~ ERROR static of uninhabited type
//~| WARN: previously accepted
//~| ERROR could not evaluate static initializer
static NEVER2: Void = unsafe { std::mem::transmute(()) }; //~ ERROR static of uninhabited type
//~| WARN: previously accepted
//~| ERROR could not evaluate static initializer

fn main() {}
