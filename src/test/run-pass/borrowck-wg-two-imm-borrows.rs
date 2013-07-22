// Test that we can borrow the same @mut box twice, so long as both are imm.

fn add(x:&int, y:&int)
{
    *x + *y;
}

pub fn main()
{
    let z = @mut [1,2,3];
    let z2 = z;
    add(&z[0], &z2[0]);
    printfln!("%d", z[0]);
}
