struct Foo {
    x: i32
}

impl Foo {
    fn foo(&self) {
        let _ = format!("{x}"); //~ ERROR cannot find value `x` in this scope [E0425]
        let _ = format!("{x }"); //~ ERROR cannot find value `x` in this scope [E0425]
        let _ = format!("{ x}"); //~ ERROR invalid format string: expected `}`, found `x`
        let _ = format!("{}", x); //~ ERROR cannot find value `x` in this scope [E0425]
        println!("{x}"); //~ ERROR cannot find value `x` in this scope [E0425]
    }
}

fn main(){}
