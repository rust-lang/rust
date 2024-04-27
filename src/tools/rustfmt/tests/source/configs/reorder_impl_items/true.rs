// rustfmt-reorder_impl_items: true

struct Dummy;

impl Iterator for Dummy {
    fn next(&mut self) -> Option<Self::Item> {
        None
    }

    type Item = i32;
}
