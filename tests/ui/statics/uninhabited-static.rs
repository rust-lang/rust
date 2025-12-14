#![feature(never_type)]

enum Void {}
extern "C" {
    static VOID: Void; //~ ERROR static of uninhabited type
    static NEVER: !; //~ ERROR static of uninhabited type
}

static VOID2: Void = unsafe { std::mem::transmute(()) };
//~^ ERROR static of uninhabited type
//~| ERROR value of uninhabited type `Void`
static NEVER2: Void = unsafe { std::mem::transmute(()) };
//~^ ERROR static of uninhabited type
//~| ERROR value of uninhabited type `Void`

fn main() {}
