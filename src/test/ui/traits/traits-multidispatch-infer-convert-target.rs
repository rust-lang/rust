// run-pass
// Test that we can infer the Target based on the Self or vice versa.


use std::mem;

trait Convert<Target> {
    fn convert(&self) -> Target;
}

impl Convert<u32> for i16 {
    fn convert(&self) -> u32 {
        *self as u32
    }
}

impl Convert<i16> for u32 {
    fn convert(&self) -> i16 {
        *self as i16
    }
}

fn test<T,U>(_: T, _: U, t_size: usize, u_size: usize)
where T : Convert<U>
{
    assert_eq!(mem::size_of::<T>(), t_size);
    assert_eq!(mem::size_of::<U>(), u_size);
}

fn main() {
    // T = i16, U = u32
    test(22_i16, Default::default(),  2, 4);

    // T = u32, U = i16
    test(22_u32, Default::default(), 4, 2);
}
