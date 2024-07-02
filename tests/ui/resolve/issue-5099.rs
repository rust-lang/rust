trait B <A> {
    fn a() -> A {
        this.a //~ ERROR cannot find value `this`
    }
    fn b(x: i32) {
        this.b(x); //~ ERROR cannot find value `this`
    }
    fn c() {
        let _ = || this.a; //~ ERROR cannot find value `this`
    }
}

fn main() {}
