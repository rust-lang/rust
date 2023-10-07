//@no-rustfix
#![warn(clippy::iter_without_into_iter)]

fn main() {
    {
        struct S;
        impl S {
            pub fn iter(&self) -> std::slice::Iter<'_, u8> {
                //~^ ERROR: `iter` method without an `IntoIterator` impl
                [].iter()
            }
            pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, u8> {
                //~^ ERROR: `iter_mut` method without an `IntoIterator` impl
                [].iter_mut()
            }
        }
    }
    {
        struct S;
        impl S {
            pub fn iter(&self) -> impl Iterator<Item = &u8> {
                // RPITIT is not stable, so we can't generally suggest it here yet
                [].iter()
            }
        }
    }
    {
        struct S<'a>(&'a mut [u8]);
        impl<'a> S<'a> {
            pub fn iter(&self) -> std::slice::Iter<'_, u8> {
                //~^ ERROR: `iter` method without an `IntoIterator` impl
                self.0.iter()
            }
            pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, u8> {
                //~^ ERROR: `iter_mut` method without an `IntoIterator` impl
                self.0.iter_mut()
            }
        }
    }
    {
        // Incompatible signatures
        struct S;
        impl S {
            pub fn iter(self) -> std::slice::Iter<'static, u8> {
                todo!()
            }
        }
        struct S2;
        impl S2 {
            pub async fn iter(&self) -> std::slice::Iter<'static, u8> {
                todo!()
            }
        }
        struct S3;
        impl S3 {
            pub fn iter(&self, _additional_param: ()) -> std::slice::Iter<'static, u8> {
                todo!()
            }
        }
        struct S4<T>(T);
        impl<T> S4<T> {
            pub fn iter<U>(&self) -> std::slice::Iter<'static, (T, U)> {
                todo!()
            }
        }
        struct S5<T>(T);
        impl<T> S5<T> {
            pub fn iter(&self) -> std::slice::Iter<'static, T> {
                todo!()
            }
        }
    }
    {
        struct S<T>(T);
        impl<T> S<T> {
            pub fn iter(&self) -> std::slice::Iter<'_, T> {
                //~^ ERROR: `iter` method without an `IntoIterator` impl
                todo!()
            }
            pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
                //~^ ERROR: `iter_mut` method without an `IntoIterator` impl
                todo!()
            }
        }
    }
    {
        struct S<T>(T);
        impl<T> S<T> {
            pub fn iter(&self) -> std::slice::Iter<'_, T> {
                // Don't lint, there's an existing (wrong) IntoIterator impl
                todo!()
            }
        }

        impl<'a, T> IntoIterator for &'a S<T> {
            type Item = &'a String;
            type IntoIter = std::slice::Iter<'a, String>;
            fn into_iter(self) -> Self::IntoIter {
                todo!()
            }
        }
    }
    {
        struct S<T>(T);
        impl<T> S<T> {
            pub fn iter_mut(&self) -> std::slice::IterMut<'_, T> {
                // Don't lint, there's an existing (wrong) IntoIterator impl
                todo!()
            }
        }

        impl<'a, T> IntoIterator for &'a mut S<T> {
            type Item = &'a mut String;
            type IntoIter = std::slice::IterMut<'a, String>;
            fn into_iter(self) -> Self::IntoIter {
                todo!()
            }
        }
    }
}
