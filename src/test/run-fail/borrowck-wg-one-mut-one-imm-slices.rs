// error-pattern:borrowed

// Test that write guards trigger when arguments are coerced to slices.

fn add(x:&mut [int], y:&[int])
{
    x[0] = x[0] + y[0];
}

pub fn main()
{
    let z = @mut [1,2,3];
    let z2 = z;
    add(z, z2);
    print(fmt!("%d\n", z[0]));
}
