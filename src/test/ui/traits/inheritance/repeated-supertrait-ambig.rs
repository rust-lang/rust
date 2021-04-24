// Test a case of a trait which extends the same supertrait twice, but
// with difference type parameters. Test then that when we don't give
// enough information to pick between these, no selection is made. In
// this particular case, the two choices are i64/u64 -- so when we use
// an integer literal, we wind up falling this literal back to i32.
// See also `run-pass/trait-repeated-supertrait.rs`.

trait CompareTo<T> {
    fn same_as(&self, t: T) -> bool;
}

trait CompareToInts : CompareTo<i64> + CompareTo<u64> {
}

impl CompareTo<i64> for i64 {
    fn same_as(&self, t: i64) -> bool { *self == t }
}

impl CompareTo<u64> for i64 {
    fn same_as(&self, t: u64) -> bool { *self == (t as i64) }
}

impl CompareToInts for i64 { }

fn with_obj(c: &dyn CompareToInts) -> bool {
    c.same_as(22) //~ ERROR `dyn CompareToInts: CompareTo<i32>` is not satisfied
}

fn with_trait<C:CompareToInts>(c: &C) -> bool {
    c.same_as(22) //~ ERROR `C: CompareTo<i32>` is not satisfied
}

fn with_ufcs1<C:CompareToInts>(c: &C) -> bool {
    <dyn CompareToInts>::same_as(c, 22) //~ ERROR `dyn CompareToInts: CompareTo<i32>` is not satisfi
}

fn with_ufcs2<C:CompareToInts>(c: &C) -> bool {
    CompareTo::same_as(c, 22) //~ ERROR `C: CompareTo<i32>` is not satisfied
}

fn main() {
    assert_eq!(22_i64.same_as(22), true); //~ ERROR `i64: CompareTo<i32>` is not satisfied
}
