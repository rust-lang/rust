// error-pattern:borrowed

// Test that arguments trigger when there are *two mutable* borrows
// of indices.

fn add(x:&mut int, y:&mut int)
{
    *x = *x + *y;
}

pub fn main()
{
    let z = @mut [1,2,3];
    let z2 = z;
    add(&mut z[0], &mut z2[0]);
    printfln!("%d", z[0]);
}
