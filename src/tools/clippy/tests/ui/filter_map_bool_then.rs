//@aux-build:proc_macros.rs
#![allow(
    clippy::clone_on_copy,
    clippy::map_identity,
    clippy::unnecessary_lazy_evaluations,
    clippy::unnecessary_filter_map,
    unused
)]
#![warn(clippy::filter_map_bool_then)]

#[macro_use]
extern crate proc_macros;

#[derive(Clone, PartialEq)]
struct NonCopy;

fn main() {
    let v = vec![1, 2, 3, 4, 5, 6];
    v.clone().iter().filter_map(|i| (i % 2 == 0).then(|| i + 1));
    v.clone().into_iter().filter_map(|i| (i % 2 == 0).then(|| i + 1));
    v.clone()
        .into_iter()
        .filter_map(|i| -> Option<_> { (i % 2 == 0).then(|| i + 1) });
    v.clone()
        .into_iter()
        .filter(|&i| i != 1000)
        .filter_map(|i| (i % 2 == 0).then(|| i + 1));
    v.iter()
        .copied()
        .filter(|&i| i != 1000)
        .filter_map(|i| (i.clone() % 2 == 0).then(|| i + 1));
    // Despite this is non-copy, `is_copy` still returns true (at least now) because it's `&NonCopy`,
    // and any `&` is `Copy`. So since we can dereference it in `filter` (since it's then `&&NonCopy`),
    // we can lint this and still get the same input type.
    // See: <https://doc.rust-lang.org/std/primitive.reference.html#trait-implementations-1>
    let v = vec![NonCopy, NonCopy];
    v.clone().iter().filter_map(|i| (i == &NonCopy).then(|| i));
    // Do not lint
    let v = vec![NonCopy, NonCopy];
    v.clone().into_iter().filter_map(|i| (i == NonCopy).then(|| i));
    // `&mut` is `!Copy`.
    let v = vec![NonCopy, NonCopy];
    v.clone().iter_mut().filter_map(|i| (i == &mut NonCopy).then(|| i));
    external! {
        let v = vec![1, 2, 3, 4, 5, 6];
        v.clone().into_iter().filter_map(|i| (i % 2 == 0).then(|| i + 1));
    }
    with_span! {
        span
        let v = vec![1, 2, 3, 4, 5, 6];
        v.clone().into_iter().filter_map(|i| (i % 2 == 0).then(|| i + 1));
    }
}

fn issue11309<'a>(iter: impl Iterator<Item = (&'a str, &'a str)>) -> Vec<&'a str> {
    iter.filter_map(|(_, s): (&str, _)| Some(s)).collect()
}

fn issue11503() {
    let bools: &[bool] = &[true, false, false, true];
    let _: Vec<usize> = bools.iter().enumerate().filter_map(|(i, b)| b.then(|| i)).collect();

    // Need to insert multiple derefs if there is more than one layer of references
    let bools: &[&&bool] = &[&&true, &&false, &&false, &&true];
    let _: Vec<usize> = bools.iter().enumerate().filter_map(|(i, b)| b.then(|| i)).collect();

    // Should also suggest derefs when going through a mutable reference
    let bools: &[&mut bool] = &[&mut true];
    let _: Vec<usize> = bools.iter().enumerate().filter_map(|(i, b)| b.then(|| i)).collect();

    // Should also suggest derefs when going through a custom deref
    struct DerefToBool;
    impl std::ops::Deref for DerefToBool {
        type Target = bool;
        fn deref(&self) -> &Self::Target {
            &true
        }
    }
    let bools: &[&&DerefToBool] = &[&&DerefToBool];
    let _: Vec<usize> = bools.iter().enumerate().filter_map(|(i, b)| b.then(|| i)).collect();
}
