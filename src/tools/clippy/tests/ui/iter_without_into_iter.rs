//@no-rustfix
//@aux-build:proc_macros.rs
#![warn(clippy::iter_without_into_iter)]
#![allow(clippy::needless_lifetimes)]
extern crate proc_macros;

pub struct S1;
impl S1 {
    pub fn iter(&self) -> std::slice::Iter<'_, u8> {
        //~^ iter_without_into_iter
        [].iter()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, u8> {
        //~^ iter_without_into_iter
        [].iter_mut()
    }
}

pub struct S2;
impl S2 {
    pub fn iter(&self) -> impl Iterator<Item = &u8> {
        // RPITIT is not stable, so we can't generally suggest it here yet
        [].iter()
    }
}

pub struct S3<'a>(&'a mut [u8]);
impl<'a> S3<'a> {
    pub fn iter(&self) -> std::slice::Iter<'_, u8> {
        //~^ iter_without_into_iter
        self.0.iter()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, u8> {
        //~^ iter_without_into_iter
        self.0.iter_mut()
    }
}

// Incompatible signatures
pub struct S4;
impl S4 {
    pub fn iter(self) -> std::slice::Iter<'static, u8> {
        todo!()
    }
}

pub struct S5;
impl S5 {
    pub async fn iter(&self) -> std::slice::Iter<'static, u8> {
        todo!()
    }
}

pub struct S6;
impl S6 {
    pub fn iter(&self, _additional_param: ()) -> std::slice::Iter<'static, u8> {
        todo!()
    }
}

pub struct S7<T>(T);
impl<T> S7<T> {
    pub fn iter<U>(&self) -> std::slice::Iter<'static, (T, U)> {
        todo!()
    }
}

pub struct S8<T>(T);
impl<T> S8<T> {
    pub fn iter(&self) -> std::slice::Iter<'static, T> {
        //~^ iter_without_into_iter
        todo!()
    }
}

// ===========================
pub struct S9<T>(T);
impl<T> S9<T> {
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        //~^ iter_without_into_iter
        todo!()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        //~^ iter_without_into_iter
        todo!()
    }
}

pub struct S10<T>(T);
impl<T> S10<T> {
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        // Don't lint, there's an existing (wrong) IntoIterator impl
        todo!()
    }
}

impl<'a, T> IntoIterator for &'a S10<T> {
    type Item = &'a String;
    type IntoIter = std::slice::Iter<'a, String>;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

pub struct S11<T>(T);
impl<T> S11<T> {
    pub fn iter_mut(&self) -> std::slice::IterMut<'_, T> {
        // Don't lint, there's an existing (wrong) IntoIterator impl
        todo!()
    }
}
impl<'a, T> IntoIterator for &'a mut S11<T> {
    type Item = &'a mut String;
    type IntoIter = std::slice::IterMut<'a, String>;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

// Private type not exported: don't lint
struct S12;
impl S12 {
    fn iter(&self) -> std::slice::Iter<'_, u8> {
        todo!()
    }
}

pub struct Issue12037;
macro_rules! generate_impl {
    () => {
        impl Issue12037 {
            fn iter(&self) -> std::slice::Iter<'_, u8> {
                //~^ iter_without_into_iter
                todo!()
            }
        }
    };
}
generate_impl!();

proc_macros::external! {
    pub struct ImplWithForeignSpan;
    impl ImplWithForeignSpan {
        fn iter(&self) -> std::slice::Iter<'_, u8> {
            todo!()
        }
    }
}

pub struct Allowed;
impl Allowed {
    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(&self) -> std::slice::Iter<'_, u8> {
        todo!()
    }
}

fn main() {}
