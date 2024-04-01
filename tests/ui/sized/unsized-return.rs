trait T {}

fn foo() -> dyn T { //~ E0746
   todo!()
}

fn main() {
    let x = foo(); //~ ERROR E0277
    let x: dyn T = foo(); //~ ERROR E0277
}
