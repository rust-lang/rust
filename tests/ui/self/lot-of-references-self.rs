struct A ;

impl A {
    fn a(&&self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, please remove extra references
    //~| HELP  or if you want exactly
    fn b(&&&&&&self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, please remove extra references
    //~| HELP  or if you want exactly
    fn c(&self) {}
    fn d(&mut &self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, please remove extra references
    //~| HELP  or if you want exactly
    fn e(&mut &&&self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, please remove extra references
    //~| HELP  or if you want exactly
    fn f(&mut &mut &mut self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, please remove extra references
    //~| HELP  or if you want exactly
    fn g(&mut & &mut self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, please remove extra references
    //~| HELP  or if you want exactly
    fn h(&mut & & & && &        & self) {}
    //~^ ERROR expected one of
    //~| HELP `self` should be `self`, `&self` or `&mut self`, please remove extra references
    //~| HELP  or if you want exactly
}

fn main() {}
