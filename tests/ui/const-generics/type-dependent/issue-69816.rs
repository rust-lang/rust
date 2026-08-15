//@ run-pass
trait IterExt: Sized + Iterator {
    fn default_for_size<const N: usize>(self) -> [Self::Item; N]
    where
        [Self::Item; N]: Default,
    {
        Default::default()
    }
}

impl<T: Iterator> IterExt for T {}

fn main(){
    const N: usize = 10;
    let arr = (0u32..10).default_for_size::<N>();
    assert_eq!(arr, [0; 10]);
}
