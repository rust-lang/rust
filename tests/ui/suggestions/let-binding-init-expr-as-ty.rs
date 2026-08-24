fn foo(num: i32) -> i32 {
    // FIXME: This case doesn't really check that `from_be` is a valid function in `i32`.
    let foo: i32::from_be(num);
    //~^ ERROR cannot find type `num` in this scope
    //~| ERROR expected type, found associated function call
    foo
}

struct S;

impl S {
    fn new(_: ()) -> S {
        S
    }
}

// We should still mention that it should be `S::new(..)`, even though rtn is not allowed there:
struct K(S::new(())); //~ ERROR return type notation not allowed in this position yet

fn bar() {}

fn main() {
    let _ = foo(42);
    // Associated functions (#134087)
    let x: Vec::new(); //~ ERROR expected type, found associated function call
    let x: Vec<()>::new(); //~ ERROR expected type, found associated function call
    let x: S::new(..); //~ ERROR expected type, found associated function call
    //~^ ERROR return type notation is experimental
    let x: S::new(()); //~ ERROR expected type, found associated function call

    // Literals
    let x: 42; //~ ERROR expected type, found `42`
    let x: ""; //~ ERROR expected type, found `""`

    // Functions
    let x: bar(); //~ ERROR cannot find type `bar` in this scope
    let x: bar; //~ ERROR cannot find type `bar` in this scope

    // Locals
    let x: x; //~ ERROR cannot find type `x` in this scope
}
