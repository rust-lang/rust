#![feature(never_type)]
#![deny(uninhabited_static)]

enum Void {}
extern {
    static VOID: Void; //~ ERROR static of uninhabited type
    //~| WARN: previously accepted
    static NEVER: !; //~ ERROR static of uninhabited type
    //~| WARN: previously accepted
}

fn main() {}
