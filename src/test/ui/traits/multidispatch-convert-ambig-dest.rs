// Check that we get an error in a multidisptach scenario where the
// set of impls is ambiguous.

trait Convert<Target> {
    fn convert(&self) -> Target;
}

impl Convert<i8> for i32 {
    fn convert(&self) -> i8 {
        *self as i8
    }
}

impl Convert<i16> for i32 {
    fn convert(&self) -> i16 {
        *self as i16
    }
}

fn test<T,U>(_: T, _: U)
where T : Convert<U>
{
}

fn a() {
    test(22, std::default::Default::default());
    //~^ ERROR type annotations needed [E0282]
}

fn main() {}
