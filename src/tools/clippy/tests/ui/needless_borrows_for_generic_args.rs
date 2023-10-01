#![warn(clippy::needless_borrows_for_generic_args)]
#![allow(
    clippy::unnecessary_to_owned,
    clippy::unnecessary_literal_unwrap,
    clippy::needless_borrow
)]

use core::ops::Deref;
use std::any::Any;
use std::ffi::OsStr;
use std::fmt::{Debug, Display};
use std::path::Path;
use std::process::Command;

fn main() {
    let _ = Command::new("ls").args(&["-a", "-l"]).status().unwrap();
    let _ = Path::new(".").join(&&".");
    let _ = Any::type_id(&""); // Don't lint. `Any` is only bound
    let _ = Box::new(&""); // Don't lint. Type parameter appears in return type
    let _ = Some("").unwrap_or(&"");
    let _ = std::fs::write("x", &"".to_string());

    {
        #[derive(Clone, Copy)]
        struct X;

        impl Deref for X {
            type Target = X;
            fn deref(&self) -> &Self::Target {
                self
            }
        }

        fn deref_target_is_x<T: Deref<Target = X>>(_: T) {}

        deref_target_is_x(&X);
    }
    {
        fn multiple_constraints<T, U, V, X, Y>(_: T)
        where
            T: IntoIterator<Item = U> + IntoIterator<Item = X>,
            U: IntoIterator<Item = V>,
            V: AsRef<str>,
            X: IntoIterator<Item = Y>,
            Y: AsRef<OsStr>,
        {
        }

        multiple_constraints(&[[""]]);
    }
    {
        #[derive(Clone, Copy)]
        struct X;

        impl Deref for X {
            type Target = X;
            fn deref(&self) -> &Self::Target {
                self
            }
        }

        fn multiple_constraints_normalizes_to_same<T, U, V>(_: T, _: V)
        where
            T: Deref<Target = U>,
            U: Deref<Target = V>,
        {
        }

        multiple_constraints_normalizes_to_same(&X, X);
    }
    {
        fn only_sized<T>(_: T) {}
        only_sized(&""); // Don't lint. `Sized` is only bound
    }
    {
        fn ref_as_ref_path<T: 'static>(_: &'static T)
        where
            &'static T: AsRef<Path>,
        {
        }

        ref_as_ref_path(&""); // Don't lint. Argument type is not a type parameter
    }
    {
        trait RefsOnly {
            type Referent;
        }

        impl<T> RefsOnly for &T {
            type Referent = T;
        }

        fn refs_only<T, U>(_: T)
        where
            T: RefsOnly<Referent = U>,
        {
        }

        refs_only(&()); // Don't lint. `&T` implements trait, but `T` doesn't
    }
    {
        fn multiple_constraints_normalizes_to_different<T, U, V>(_: T, _: U)
        where
            T: IntoIterator<Item = U>,
            U: IntoIterator<Item = V>,
            V: AsRef<str>,
        {
        }
        multiple_constraints_normalizes_to_different(&[[""]], &[""]); // Don't lint. Projected type appears in arguments
    }
    // https://github.com/rust-lang/rust-clippy/pull/9136#pullrequestreview-1037379321
    {
        #[derive(Clone, Copy)]
        struct Iter;
        impl Iterator for Iter {
            type Item = ();
            fn next(&mut self) -> Option<Self::Item> {
                None
            }
        }
        fn takes_iter(_: impl Iterator) {}
        fn dont_warn(mut x: Iter) {
            takes_iter(&mut x);
        }
        #[allow(unused_mut)]
        fn warn(mut x: &mut Iter) {
            takes_iter(&mut x)
        }
    }
    #[clippy::msrv = "1.52.0"]
    {
        let _ = Command::new("ls").args(&["-a", "-l"]).status().unwrap();
    };
    #[clippy::msrv = "1.53.0"]
    {
        let _ = Command::new("ls").args(&["-a", "-l"]).status().unwrap();
    };
    {
        let env = "env".to_owned();
        let arg = "arg".to_owned();
        let f = |arg| {
            let loc = "loc".to_owned();
            let _ = std::fs::write("x", &env); // Don't lint. In environment
            let _ = std::fs::write("x", &arg);
            let _ = std::fs::write("x", &loc);
        };
        let _ = std::fs::write("x", &env); // Don't lint. Borrowed by `f`
        f(arg);
    }
    {
        #[derive(Debug)]
        struct X;

        impl Drop for X {
            fn drop(&mut self) {}
        }

        fn f(_: impl Debug) {}

        let x = X;
        f(&x); // Don't lint. Has significant drop
    }
    {
        fn f(_: impl AsRef<str>) {}

        let x = String::new();
        f(&x);
    }
    {
        fn f(_: impl AsRef<str>) {}
        fn f2(_: impl AsRef<str>) {}

        let x = String::new();
        f(&x);
        f2(&x);
    }
    // https://github.com/rust-lang/rust-clippy/issues/9111#issuecomment-1277114280
    // issue 9111
    {
        struct A;

        impl Extend<u8> for A {
            fn extend<T: IntoIterator<Item = u8>>(&mut self, _: T) {
                unimplemented!()
            }
        }

        impl<'a> Extend<&'a u8> for A {
            fn extend<T: IntoIterator<Item = &'a u8>>(&mut self, _: T) {
                unimplemented!()
            }
        }

        let mut a = A;
        a.extend(&[]); // vs a.extend([]);
    }
    // issue 9710
    {
        fn f(_: impl AsRef<str>) {}

        let x = String::new();
        for _ in 0..10 {
            f(&x);
        }
    }
    // issue 9739
    {
        fn foo<D: Display>(_it: impl IntoIterator<Item = D>) {}
        foo(if std::env::var_os("HI").is_some() {
            &[0]
        } else {
            &[] as &[u32]
        });
    }
    {
        struct S;

        impl S {
            fn foo<D: Display>(&self, _it: impl IntoIterator<Item = D>) {}
        }

        S.foo(if std::env::var_os("HI").is_some() {
            &[0]
        } else {
            &[] as &[u32]
        });
    }
    // issue 9782
    {
        fn foo<T: AsRef<[u8]>>(t: T) {
            println!("{}", std::mem::size_of::<T>());
            let _t: &[u8] = t.as_ref();
        }

        let a: [u8; 100] = [0u8; 100];

        // 100
        foo::<[u8; 100]>(a);
        foo(a);

        // 16
        foo::<&[u8]>(&a);
        foo(a.as_slice());

        // 8
        foo::<&[u8; 100]>(&a);
        foo(&a);
    }
    {
        struct S;

        impl S {
            fn foo<T: AsRef<[u8]>>(t: T) {
                println!("{}", std::mem::size_of::<T>());
                let _t: &[u8] = t.as_ref();
            }
        }

        let a: [u8; 100] = [0u8; 100];
        S::foo::<&[u8; 100]>(&a);
    }
    {
        struct S;

        impl S {
            fn foo<T: AsRef<[u8]>>(&self, t: T) {
                println!("{}", std::mem::size_of::<T>());
                let _t: &[u8] = t.as_ref();
            }
        }

        let a: [u8; 100] = [0u8; 100];
        S.foo::<&[u8; 100]>(&a);
    }
    // issue 10535
    {
        static SOME_STATIC: String = String::new();

        static UNIT: () = compute(&SOME_STATIC);

        pub const fn compute<T>(_: T)
        where
            T: Copy,
        {
        }
    }
}
