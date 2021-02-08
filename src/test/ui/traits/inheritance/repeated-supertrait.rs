// run-pass
// Test a case of a trait which extends the same supertrait twice, but
// with difference type parameters. Test that we can invoke the
// various methods in various ways successfully.
// See also `ui/traits/trait-repeated-supertrait-ambig.rs`.


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
    c.same_as(22_i64) && c.same_as(22_u64)
}

fn with_trait<C:CompareToInts>(c: &C) -> bool {
    c.same_as(22_i64) && c.same_as(22_u64)
}

fn with_ufcs1<C:CompareToInts>(c: &C) -> bool {
    CompareToInts::same_as(c, 22_i64) && CompareToInts::same_as(c, 22_u64)
}

fn with_ufcs2<C:CompareToInts>(c: &C) -> bool {
    CompareTo::same_as(c, 22_i64) && CompareTo::same_as(c, 22_u64)
}

fn main() {
    assert_eq!(22_i64.same_as(22_i64), true);
    assert_eq!(22_i64.same_as(22_u64), true);
    assert_eq!(with_trait(&22), true);
    assert_eq!(with_obj(&22), true);
    assert_eq!(with_ufcs1(&22), true);
    assert_eq!(with_ufcs2(&22), true);
}
