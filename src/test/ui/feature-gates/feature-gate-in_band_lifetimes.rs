#![allow(warnings)]

fn foo(x: &'x u8) -> &'x u8 { x }
//~^ ERROR use of undeclared lifetime name
//~^^ ERROR use of undeclared lifetime name

struct X<'a>(&'a u8);

impl<'a> X<'a> {
    fn inner(&self) -> &'a u8 {
        self.0
    }
}

impl<'a> X<'b> {
//~^ ERROR use of undeclared lifetime name
    fn inner_2(&self) -> &'b u8 {
    //~^ ERROR use of undeclared lifetime name
        self.0
    }
}

impl X<'b> {
//~^ ERROR use of undeclared lifetime name
    fn inner_3(&self) -> &'b u8 {
    //~^ ERROR use of undeclared lifetime name
        self.0
    }
}

struct Y<T>(T);

impl Y<&'a u8> {
    //~^ ERROR use of undeclared lifetime name
    fn inner(&self) -> &'a u8 {
    //~^ ERROR use of undeclared lifetime name
        self.0
    }
}

trait MyTrait<'a> {
    fn my_lifetime(&self) -> &'a u8;
    fn any_lifetime() -> &'b u8;
    //~^ ERROR use of undeclared lifetime name
    fn borrowed_lifetime(&'b self) -> &'b u8;
    //~^ ERROR use of undeclared lifetime name
    //~^^ ERROR use of undeclared lifetime name
}

impl MyTrait<'a> for Y<&'a u8> {
//~^ ERROR use of undeclared lifetime name
//~^^ ERROR use of undeclared lifetime name
    fn my_lifetime(&self) -> &'a u8 { self.0 }
    //~^ ERROR use of undeclared lifetime name
    fn any_lifetime() -> &'b u8 { &0 }
    //~^ ERROR use of undeclared lifetime name
    fn borrowed_lifetime(&'b self) -> &'b u8 { &*self.0 }
    //~^ ERROR use of undeclared lifetime name
    //~^^ ERROR use of undeclared lifetime name
}

fn main() {}
