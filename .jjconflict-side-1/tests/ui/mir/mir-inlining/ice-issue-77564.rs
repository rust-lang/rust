//@ run-pass
//@ compile-flags:-Zmir-opt-level=3

use std::mem::MaybeUninit;
const N: usize = 2;

trait CollectArray<A>: Iterator<Item = A> {
    fn inner_array(&mut self) -> [A; N];
    fn collect_array(&mut self) -> [A; N] {
        let result = self.inner_array();
        assert!(self.next().is_none());
        result
    }
}

impl<A, I: ?Sized> CollectArray<A> for I
where
    I: Iterator<Item = A>,
{
    fn inner_array(&mut self) -> [A; N] {
        let mut result: [MaybeUninit<A>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        for (dest, item) in result.iter_mut().zip(self) {
            *dest = MaybeUninit::new(item);
        }
        let temp_ptr: *const [MaybeUninit<A>; N] = &result;
        unsafe { std::ptr::read(temp_ptr as *const [A; N]) }
    }
}

fn main() {
    assert_eq!(
        [[1, 2], [3, 4]]
            .iter()
            .map(|row| row.iter().collect_array())
            .collect_array(),
        [[&1, &2], [&3, &4]]
    );
}
