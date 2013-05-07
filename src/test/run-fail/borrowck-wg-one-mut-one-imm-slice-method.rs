// error-pattern:borrowed

// Test that write guards trigger when there is a coercion to
// a slice on the receiver of a method.

trait MyMutSlice {
    fn my_mut_slice(self) -> Self;
}

impl<'self, T> MyMutSlice for &'self mut [T] {
    fn my_mut_slice(self) -> &'self mut [T] {
        self
    }
}

trait MySlice {
    fn my_slice(self) -> Self;
}

impl<'self, T> MySlice for &'self [T] {
    fn my_slice(self) -> &'self [T] {
        self
    }
}

fn add(x:&mut [int], y:&[int])
{
    x[0] = x[0] + y[0];
}

pub fn main()
{
    let z = @mut [1,2,3];
    let z2 = z;
    add(z.my_mut_slice(), z2.my_slice());
    print(fmt!("%d\n", z[0]));
}
