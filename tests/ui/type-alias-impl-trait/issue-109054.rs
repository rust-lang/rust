//@ edition:2021

#![feature(type_alias_impl_trait)]

struct CallMe;

type ReturnType<'a> = impl std::future::Future<Output = u32> + 'a;
type FnType = impl Fn(&u32) -> ReturnType;

impl std::ops::Deref for CallMe {
    type Target = FnType;

    #[define_opaque(FnType)]
    fn deref(&self) -> &Self::Target {
        //~^ ERROR: item does not constrain `ReturnType
        #[define_opaque(ReturnType)]
        fn inner(val: &u32) -> ReturnType {
            async move { *val * 2 }
        }

        &inner
    }
}

fn main() {}
