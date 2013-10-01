// error-pattern:borrowed

// Test that write guards trigger when we are indexing into
// an @mut vector.

fn add(x:&mut int, y:&int)
{
    *x = *x + *y;
}

pub fn main()
{
    let z = @mut [1,2,3];
    let z2 = z;
    add(&mut z[0], &z2[0]);
    println!("{}", z[0]);
}
