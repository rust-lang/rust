#![feature(optin_builtin_traits)]

struct Foo;
impl !Sync for Foo {}

struct Bar(i32);
impl !Sync for Bar {}

const FOO : &Foo = &Foo; //~ ERROR cannot borrow
const FOO2 : Option<&Foo> = Some(&Foo); //~ ERROR cannot borrow
//~^ ERROR borrowed value does not live long enough
const FOO3 : &Option<Foo> = &Some(Foo); //~ ERROR cannot borrow

const BAR : &Bar = &Bar(42); //~ ERROR cannot borrow
const BAR2 : Option<&Bar> = Some(&Bar(42)); //~ ERROR cannot borrow
//~^ ERROR borrowed value does not live long enough
const BAR3 : &Option<Bar> = &Some(Bar(42)); //~ ERROR cannot borrow

fn main() {}
