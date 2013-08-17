// error-pattern:borrowed

// Test that if you mut borrow then imm borrow it fails.

fn add1(a:@mut int)
{
    add2(a); // already frozen
}

fn add2(_:&int)
{
}

pub fn main()
{
    let a = @mut 3;
    let _b = &mut *a; // freezes a
    add1(a);
}
