// error-pattern:borrowed

// Test that if you imm borrow then mut borrow it fails.

fn add1(a:@mut int)
{
    add2(a); // already frozen
}

fn add2(_:&mut int)
{
}

pub fn main()
{
    let a = @mut 3;
    let _b = &*a; // freezes a
    add1(a);
}
