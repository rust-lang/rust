//@ run-pass
trait T<const A: usize> {
    fn l<const N: bool>() -> usize;
    fn r<const N: bool>() -> bool;
}

struct S;

impl<const N: usize> T<N> for S {
    fn l<const M: bool>() -> usize { N }
    fn r<const M: bool>() -> bool { M }
}

fn main() {
   assert_eq!(<S as T<123>>::l::<true>(), 123);
   assert!(<S as T<123>>::r::<true>());
}
