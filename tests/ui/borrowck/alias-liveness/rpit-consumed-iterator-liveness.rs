//@ edition: 2024
//@ check-pass

// Minimized from polars-async's `parallelize_first_to_local` and polars-stream's
// `rechunk_par` crater regressions in #161154.

#![forbid(unsafe_code)]

mod consumed_input {
    fn first_to_local<'o, I>(iter: I) -> impl Iterator<Item = u32> + 'o
    where
        I: Iterator<Item = u32>,
    {
        let v: Vec<u32> = iter.collect();
        v.into_iter()
    }

    pub fn through_a_slice(columns: &mut [u32]) {
        for x in first_to_local(columns.iter_mut().map(|c| *c)) {
            columns[0] = x;
        }
    }

    pub fn through_a_local() {
        let mut cells = vec![1u32, 2, 3];
        let it = first_to_local(cells.iter_mut().map(|c| *c));
        cells[0] = 7;
        drop(it);
    }

    fn no_type_arg<'o>(v: Vec<u32>) -> impl Iterator<Item = u32> + 'o {
        v.into_iter()
    }

    pub fn without_a_type_arg(columns: &mut [u32]) {
        for x in no_type_arg(columns.iter_mut().map(|c| *c).collect()) {
            columns[0] = x;
        }
    }
}

// We shouldn't look directly at the return type, but at alias inside it.
mod wrapped_return {
    fn tuple<'o, I: Iterator<Item = u32>>(iter: I) -> (impl Iterator<Item = u32> + 'o,) {
        (iter.collect::<Vec<_>>().into_iter(),)
    }

    fn optional<'o, I: Iterator<Item = u32>>(iter: I) -> Option<impl Iterator<Item = u32> + 'o> {
        Some(iter.collect::<Vec<_>>().into_iter())
    }

    pub fn test_tuple(columns: &mut [u32]) {
        for x in tuple(columns.iter_mut().map(|c| *c)).0 {
            columns[0] = x;
        }
    }

    pub fn test_optional(columns: &mut [u32]) {
        for x in optional(columns.iter_mut().map(|c| *c)).unwrap() {
            columns[0] = x;
        }
    }
}

// We don't need to treat `F` as live because it is `'static`
mod nested_static_item {
    fn first_to_local<'i, 'o, I, F>(iter: I) -> impl Iterator<Item = impl Copy + 'static> + 'o
    where
        I: Iterator<Item = F> + 'i,
        F: Copy + 'static,
    {
        iter.collect::<Vec<F>>().into_iter()
    }

    pub fn through_a_slice(columns: &mut [u32]) {
        for _ in first_to_local(columns.iter_mut().map(|c| *c)) {
            columns[0] = 1;
        }
    }
}

// Item bounds can add `'static`
mod static_item_bound {
    use std::any::Any;

    trait Items {
        type Item: 'static;
        fn into_item(self) -> Self::Item;
    }

    impl Items for &mut u32 {
        type Item = u32;
        fn into_item(self) -> u32 {
            *self
        }
    }

    fn first_to_local<'i, 'o, I: Items + 'i>(item: I) -> impl Iterator<Item = Box<dyn Any>> + 'o {
        std::iter::once(Box::new(item.into_item()) as Box<dyn Any>)
    }

    pub fn through_a_slice(columns: &mut [u32]) {
        for _ in first_to_local(&mut columns[0]) {
            columns[0] = 1;
        }
    }
}

mod static_region_in_trait_bound {
    trait Tagged<'r> {}
    impl<'r, T> Tagged<'r> for T {}

    fn first_to_local<'o, I>(iter: I) -> impl Iterator<Item = u32> + 'o
    where
        I: Iterator<Item = u32> + Tagged<'static>,
    {
        iter.collect::<Vec<_>>().into_iter()
    }

    pub fn through_a_slice(columns: &mut [u32]) {
        for x in first_to_local(columns.iter_mut().map(|c| *c)) {
            columns[0] = x;
        }
    }
}

fn main() {}
