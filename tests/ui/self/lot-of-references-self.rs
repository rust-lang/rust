struct A ;

impl A {
    fn a(&&self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, consider removing extra references
    fn b(&&&&&&self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, consider removing extra references
    fn c(&self) {}
    fn d(&mut &self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, consider removing extra references
    fn e(&mut &&&self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, consider removing extra references
    fn f(&mut &mut &mut self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, consider removing extra references
    fn g(&mut & &mut self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, consider removing extra references
    fn h(&mut & & & && &        & self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, consider removing extra references
}

fn main() {}
