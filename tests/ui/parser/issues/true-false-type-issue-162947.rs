struct A;

impl A {
    fn _a() -> true { //~ ERROR: expected type, found keyword `true`
        false
    }
    fn b(&self) {}
}

fn main() {
    let a = A;
    a.b(); //~ ERROR E0599

    let _b: true = true; //~ ERROR: expected type, found keyword `true`
    //~^ ERROR E0070
}
