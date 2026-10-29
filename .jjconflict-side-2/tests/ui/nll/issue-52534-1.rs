struct Test;

impl Test {
    fn bar(&self, x: &u32) -> &u32 {
        let x = 22;
        &x
//~^ ERROR cannot return reference to local variable
    }
}

fn foo(x: &u32) -> &u32 {
    let x = 22;
    &x
//~^ ERROR cannot return reference to local variable
}

fn baz(x: &u32) -> &&u32 {
    let x = 22;
    &&x
//~^ ERROR cannot return value referencing local variable `x`
//~| ERROR cannot return reference to temporary value
}

fn foobazbar<'a>(x: u32, y: &'a u32) -> &'a u32 {
    let x = 22;
    &x
//~^ ERROR cannot return reference to local variable `x`
}

fn foobar<'a>(x: &'a u32) -> &'a u32 {
    let x = 22;
    &x
//~^ ERROR cannot return reference to local variable
}

fn foobaz<'a, 'b>(x: &'a u32, y: &'b u32) -> &'a u32 {
    let x = 22;
    &x
//~^ ERROR cannot return reference to local variable
}

fn foobarbaz<'a, 'b>(x: &'a u32, y: &'b u32, z: &'a u32) -> &'a u32 {
    let x = 22;
    &x
//~^ ERROR cannot return reference to local variable
}

fn main() { }
