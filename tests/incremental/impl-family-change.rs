// Changing the argument of one impl in a split family between sessions must be picked up.
//@ revisions: rpass1 rpass2

trait Tr {
    fn val() -> usize;
}
struct S<const N: usize>;

macro_rules! family {
    ($($n:literal)*) => { $(impl Tr for S<$n> { fn val() -> usize { $n } })* };
}
family!(0 1 2);
#[cfg(rpass1)]
family!(3);
#[cfg(rpass2)]
family!(7);

fn main() {
    #[cfg(rpass1)]
    assert_eq!(<S<3> as Tr>::val(), 3);
    #[cfg(rpass2)]
    assert_eq!(<S<7> as Tr>::val(), 7);
}
