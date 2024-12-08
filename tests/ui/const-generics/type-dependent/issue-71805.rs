//@ run-pass
use std::mem::MaybeUninit;

trait CollectSlice<'a>: Iterator {
    fn inner_array<const N: usize>(&mut self) -> [Self::Item; N];

    fn collect_array<const N: usize>(&mut self) -> [Self::Item; N] {
        let result = self.inner_array();
        assert!(self.next().is_none());
        result
    }
}

impl<'a, I: ?Sized> CollectSlice<'a> for I
where
    I: Iterator,
{
    fn inner_array<const N: usize>(&mut self) -> [Self::Item; N] {
        let mut result: [MaybeUninit<Self::Item>; N] =
            unsafe { MaybeUninit::uninit().assume_init() };

        let mut count = 0;
        for (dest, item) in result.iter_mut().zip(self) {
            *dest = MaybeUninit::new(item);
            count += 1;
        }

        assert_eq!(N, count);

        let temp_ptr: *const [MaybeUninit<Self::Item>; N] = &result;
        unsafe { std::ptr::read(temp_ptr as *const [Self::Item; N]) }
    }
}

fn main() {
    let mut foos = [0u64; 9].iter().cloned();
    let _bar: [u64; 9] = foos.collect_array::<9_usize>();
}
