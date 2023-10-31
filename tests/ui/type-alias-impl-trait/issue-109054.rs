// edition:2021

#![feature(type_alias_impl_trait)]

struct CallMe;

type ReturnType<'a> = impl std::future::Future<Output = u32> + 'a;
type FnType = impl Fn(&u32) -> ReturnType;

impl std::ops::Deref for CallMe {
    type Target = FnType;

    fn deref(&self) -> &Self::Target {
        fn inner(val: &u32) -> ReturnType {
            async move { *val * 2 }
        }

        &inner //~ ERROR: expected generic lifetime parameter, found `'_`
    }
}

fn main() {}
