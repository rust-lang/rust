//@no-rustfix
#![warn(clippy::into_iter_without_iter)]

use std::iter::IntoIterator;

fn main() {
    {
        struct S;

        impl<'a> IntoIterator for &'a S {
            //~^ ERROR: `IntoIterator` implemented for a reference type without an `iter` method
            type IntoIter = std::slice::Iter<'a, u8>;
            type Item = &'a u8;
            fn into_iter(self) -> Self::IntoIter {
                todo!()
            }
        }
        impl<'a> IntoIterator for &'a mut S {
            //~^ ERROR: `IntoIterator` implemented for a reference type without an `iter_mut` method
            type IntoIter = std::slice::IterMut<'a, u8>;
            type Item = &'a mut u8;
            fn into_iter(self) -> Self::IntoIter {
                todo!()
            }
        }
    }
    {
        struct S<T>(T);
        impl<'a, T> IntoIterator for &'a S<T> {
            //~^ ERROR: `IntoIterator` implemented for a reference type without an `iter` method
            type IntoIter = std::slice::Iter<'a, T>;
            type Item = &'a T;
            fn into_iter(self) -> Self::IntoIter {
                todo!()
            }
        }
        impl<'a, T> IntoIterator for &'a mut S<T> {
            //~^ ERROR: `IntoIterator` implemented for a reference type without an `iter_mut` method
            type IntoIter = std::slice::IterMut<'a, T>;
            type Item = &'a mut T;
            fn into_iter(self) -> Self::IntoIter {
                todo!()
            }
        }
    }
    {
        // Both iter and iter_mut methods exist, don't lint
        struct S<'a, T>(&'a T);

        impl<'a, T> S<'a, T> {
            fn iter(&self) -> std::slice::Iter<'a, T> {
                todo!()
            }
            fn iter_mut(&mut self) -> std::slice::IterMut<'a, T> {
                todo!()
            }
        }

        impl<'a, T> IntoIterator for &S<'a, T> {
            type IntoIter = std::slice::Iter<'a, T>;
            type Item = &'a T;
            fn into_iter(self) -> Self::IntoIter {
                todo!()
            }
        }

        impl<'a, T> IntoIterator for &mut S<'a, T> {
            type IntoIter = std::slice::IterMut<'a, T>;
            type Item = &'a mut T;
            fn into_iter(self) -> Self::IntoIter {
                todo!()
            }
        }
    }
    {
        // Only `iter` exists, no `iter_mut`
        struct S<'a, T>(&'a T);

        impl<'a, T> S<'a, T> {
            fn iter(&self) -> std::slice::Iter<'a, T> {
                todo!()
            }
        }

        impl<'a, T> IntoIterator for &S<'a, T> {
            type IntoIter = std::slice::Iter<'a, T>;
            type Item = &'a T;
            fn into_iter(self) -> Self::IntoIter {
                todo!()
            }
        }

        impl<'a, T> IntoIterator for &mut S<'a, T> {
            //~^ ERROR: `IntoIterator` implemented for a reference type without an `iter_mut` method
            type IntoIter = std::slice::IterMut<'a, T>;
            type Item = &'a mut T;
            fn into_iter(self) -> Self::IntoIter {
                todo!()
            }
        }
    }
    {
        // `iter` exists, but `IntoIterator` is implemented for an alias. inherent_impls doesn't "normalize"
        // aliases so that `inherent_impls(Alias)` where `type Alias = S` returns nothing, so this can lead
        // to fun FPs. Make sure it doesn't happen here (we're using type_of, which should skip the alias).
        struct S;

        impl S {
            fn iter(&self) -> std::slice::Iter<'static, u8> {
                todo!()
            }
        }

        type Alias = S;

        impl IntoIterator for &Alias {
            type IntoIter = std::slice::Iter<'static, u8>;
            type Item = &'static u8;
            fn into_iter(self) -> Self::IntoIter {
                todo!()
            }
        }
    }
}

fn issue11635() {
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
