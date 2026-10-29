// Regression test for #22040.

use std::fmt::Debug;

trait Expr: Debug + PartialEq {
    fn print_element_count(&self);
}

//#[derive(PartialEq)]
#[derive(Debug)]
struct SExpr<'x> {
    elements: Vec<Box<dyn Expr + 'x>>,
    //~^ ERROR E0038
}

impl<'x> PartialEq for SExpr<'x> {
    fn eq(&self, other:&SExpr<'x>) -> bool {
        println!("L1: {} L2: {}", self.elements.len(), other.elements.len());

        let result = self.elements.len() == other.elements.len();

        println!("Got compare {}", result);
        return result;
    }
}

impl <'x> SExpr<'x> {
    fn new() -> SExpr<'x> { return SExpr{elements: Vec::new(),}; }
}

impl <'x> Expr for SExpr<'x> {
    fn print_element_count(&self) {
        println!("element count: {}", self.elements.len());
    }
}

fn main() {
    let a: Box<dyn Expr> = Box::new(SExpr::new());
    //~^ ERROR: `Expr` is not dyn compatible
    let b: Box<dyn Expr> = Box::new(SExpr::new());
    //~^ ERROR: `Expr` is not dyn compatible

    // assert_eq!(a , b);
}
