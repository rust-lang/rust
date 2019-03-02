struct A;

impl A {
    fn b(self> {}
    //~^ ERROR expected one of `)`, `,`, or `:`, found `>`
}

// verify that mismatched delimiters get emitted
fn foo(] {}
//~^ ERROR incorrect close delimiter

fn main() {}
