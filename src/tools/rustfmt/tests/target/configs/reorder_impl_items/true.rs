// rustfmt-reorder_impl_items: true

struct Dummy;

impl Iterator for Dummy {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}
