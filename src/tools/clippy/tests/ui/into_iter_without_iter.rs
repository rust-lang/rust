//@no-rustfix: suggestions reference out of scope lifetimes/types
//@aux-build:proc_macros.rs
#![warn(clippy::into_iter_without_iter)]
extern crate proc_macros;

use std::iter::IntoIterator;

pub struct S1;
impl<'a> IntoIterator for &'a S1 {
    //~^ into_iter_without_iter
    type IntoIter = std::slice::Iter<'a, u8>;
    type Item = &'a u8;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}
impl<'a> IntoIterator for &'a mut S1 {
    //~^ into_iter_without_iter
    type IntoIter = std::slice::IterMut<'a, u8>;
    type Item = &'a mut u8;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

pub struct S2<T>(T);
impl<'a, T> IntoIterator for &'a S2<T> {
    //~^ into_iter_without_iter
    type IntoIter = std::slice::Iter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}
impl<'a, T> IntoIterator for &'a mut S2<T> {
    //~^ into_iter_without_iter
    type IntoIter = std::slice::IterMut<'a, T>;
    type Item = &'a mut T;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

// Both iter and iter_mut methods exist, don't lint
pub struct S3<'a, T>(&'a T);
impl<'a, T> S3<'a, T> {
    fn iter(&self) -> std::slice::Iter<'a, T> {
        todo!()
    }
    fn iter_mut(&mut self) -> std::slice::IterMut<'a, T> {
        todo!()
    }
}
impl<'a, T> IntoIterator for &S3<'a, T> {
    type IntoIter = std::slice::Iter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}
impl<'a, T> IntoIterator for &mut S3<'a, T> {
    type IntoIter = std::slice::IterMut<'a, T>;
    type Item = &'a mut T;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

// Only `iter` exists, no `iter_mut`
pub struct S4<'a, T>(&'a T);

impl<'a, T> S4<'a, T> {
    fn iter(&self) -> std::slice::Iter<'a, T> {
        todo!()
    }
}

impl<'a, T> IntoIterator for &S4<'a, T> {
    type IntoIter = std::slice::Iter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

impl<'a, T> IntoIterator for &mut S4<'a, T> {
    //~^ into_iter_without_iter
    type IntoIter = std::slice::IterMut<'a, T>;
    type Item = &'a mut T;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

// `iter` exists, but `IntoIterator` is implemented for an alias. inherent_impls doesn't "normalize"
// aliases so that `inherent_impls(Alias)` where `type Alias = S` returns nothing, so this can lead
// to fun FPs. Make sure it doesn't happen here (we're using type_of, which should skip the alias).
pub struct S5;

impl S5 {
    fn iter(&self) -> std::slice::Iter<'static, u8> {
        todo!()
    }
}

pub type Alias = S5;

impl IntoIterator for &Alias {
    type IntoIter = std::slice::Iter<'static, u8>;
    type Item = &'static u8;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

// Fine to lint, the impls comes from a local macro.
pub struct Issue12037;
macro_rules! generate_impl {
    () => {
        impl<'a> IntoIterator for &'a Issue12037 {
            //~^ into_iter_without_iter
            type IntoIter = std::slice::Iter<'a, u8>;
            type Item = &'a u8;
            fn into_iter(self) -> Self::IntoIter {
                todo!()
            }
        }
    };
}
generate_impl!();

// Impl comes from an external crate
proc_macros::external! {
    pub struct ImplWithForeignSpan;
    impl<'a> IntoIterator for &'a ImplWithForeignSpan {
        type IntoIter = std::slice::Iter<'a, u8>;
        type Item = &'a u8;
        fn into_iter(self) -> Self::IntoIter {
            todo!()
        }
    }
}

pub struct Allowed;
#[allow(clippy::into_iter_without_iter)]
impl<'a> IntoIterator for &'a Allowed {
    type IntoIter = std::slice::Iter<'a, u8>;
    type Item = &'a u8;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

fn main() {}

pub mod issue11635 {
    // A little more involved than the original repro in the issue, but this tests that it correctly
    // works for more than one deref step

    use std::ops::Deref;

    pub struct Thing(Vec<u8>);
    pub struct Thing2(Thing);

    impl Deref for Thing {
        type Target = [u8];

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl Deref for Thing2 {
        type Target = Thing;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<'a> IntoIterator for &'a Thing2 {
        type Item = &'a u8;
        type IntoIter = <&'a [u8] as IntoIterator>::IntoIter;

        fn into_iter(self) -> Self::IntoIter {
            self.0.iter()
        }
    }
}

pub mod issue12964 {
    pub struct MyIter<'a, T: 'a> {
        iter: std::slice::Iter<'a, T>,
    }

    impl<'a, T> Iterator for MyIter<'a, T> {
        type Item = &'a T;

        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next()
        }
    }

    pub struct MyContainer<T> {
        inner: Vec<T>,
    }

    impl<T> MyContainer<T> {}

    impl<T> MyContainer<T> {
        #[must_use]
        pub fn iter(&self) -> MyIter<'_, T> {
            <&Self as IntoIterator>::into_iter(self)
        }
    }

    impl<'a, T> IntoIterator for &'a MyContainer<T> {
        type Item = &'a T;

        type IntoIter = MyIter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            Self::IntoIter {
                iter: self.inner.as_slice().iter(),
            }
        }
    }
}
