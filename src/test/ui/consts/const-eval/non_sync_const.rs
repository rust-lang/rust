#![feature(optin_builtin_traits,const_let)]
use std::cell::Cell;

struct Foo;
impl !Sync for Foo {}

struct Bar(i32);
impl !Sync for Bar {}

const FOO1 : &Foo = &Foo; //~ ERROR cannot borrow
const FOO2 : Option<&Foo> = Some(&Foo); //~ ERROR cannot borrow
//~^ ERROR borrowed value does not live long enough
const FOO3 : &Option<Foo> = &Some(Foo); //~ ERROR cannot borrow

const BAR1 : &Bar = &Bar(42); //~ ERROR cannot borrow
const BAR2 : Option<&Bar> = Some(&Bar(42)); //~ ERROR cannot borrow
//~^ ERROR borrowed value does not live long enough
const BAR3 : &Option<Bar> = &Some(Bar(42)); //~ ERROR cannot borrow

/// Tests making sure that value-based reasoning does not go too far.
enum MyOption1<T> { Some(Cell<T>), None } // Never Sync, we shouldnt assume it has Sync values

trait NotSync {} // private trait!
impl NotSync for Foo {}
enum MyOption2<T: NotSync> { Some(T), None } // Never Sync, we shouldnt assume it has Sync values

const OPT1 : &MyOption1<i32> = &MyOption1::None; //~ ERROR cannot borrow
const OPT2 : &MyOption2<Foo> = &MyOption2::None; //~ ERROR cannot borrow

// Hiding in closure variables
const TESTS: &FnOnce() = &{ //~ ERROR cannot borrow
    let x = Cell::new(32);
    move || {
        x;
    }
};

fn main() {}
