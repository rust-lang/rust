//@ run-pass
// Test a case where the associated type binding (to `bool`, in this
// case) is derived from the trait definition. Issue #21636.


use std::vec;

pub trait BitIter {
    type Iter: Iterator<Item=bool>;
    fn bit_iter(self) -> <Self as BitIter>::Iter;
}

impl BitIter for Vec<bool> {
    type Iter = vec::IntoIter<bool>;
    fn bit_iter(self) -> <Self as BitIter>::Iter {
        self.into_iter()
    }
}

fn count<T>(arg: T) -> usize
    where T: BitIter
{
    let mut sum = 0;
    for i in arg.bit_iter() {
        if i {
            sum += 1;
        }
    }
    sum
}

fn main() {
    let v = vec![true, false, true];
    let c = count(v);
    assert_eq!(c, 2);
}
