//@ edition: 2021
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
//@[current] known-bug: #36262
//@[current] dont-check-compiler-stderr

// Tests that specialization does not leak into type inference.
// Regression for #36262 and duplicate issues.

#![feature(specialization)]
#![allow(incomplete_features)]

// Site 1: the receiver's type parameter, observed through a return position (#36262).
mod receiver_return {
    struct My<T>(T);

    trait Conv<T> {
        fn conv(self) -> T;
    }

    impl<T> Conv<T> for My<T> {
        default fn conv(self) -> T {
            self.0
        }
    }

    impl Conv<u32> for My<u32> {
        fn conv(self) -> u32 {
            self.0
        }
    }

    fn use_it() {
        // Should infer `i32`; the sole `My<u32>` impl steers it to `u32`.
        let x = My(0);
        let _ = x.conv() + 0i32;
    }
}

// Site 2: a method argument's trait type parameter (#91973, #38516, #67918).
mod method_arg {
    struct Foo;

    trait Bar<T> {
        fn bar(&self, _: T);
    }

    impl<T> Bar<T> for Foo {
        default fn bar(&self, _: T) {}
    }

    impl Bar<bool> for Foo {
        fn bar(&self, _: bool) {}
    }

    fn use_it() {
        // Should infer `{integer}`; the sole `Bar<bool>` impl steers it to `bool`.
        Foo.bar(42);
    }
}

// Site 3: an explicit `_` in a UFCS trait reference (#40718).
mod ufcs_infer {
    use std::vec;

    struct Foo<T>(T);

    impl<T> Foo<T> {
        fn build<I: IntoIterator<Item = T>>(it: I) -> Foo<T> {
            // The second argument should infer to `I::IntoIter`; the sole
            // `vec::IntoIter<T>` impl steers it there.
            <Self as SpecExtend<_, _>>::from_iter(it.into_iter())
        }
    }

    trait SpecExtend<T, I> {
        fn from_iter(iter: I) -> Self;
    }

    impl<T, I> SpecExtend<T, I> for Foo<T>
    where
        I: Iterator<Item = T>,
    {
        default fn from_iter(_: I) -> Self {
            panic!()
        }
    }

    impl<T> SpecExtend<T, vec::IntoIter<T>> for Foo<T> {
        fn from_iter(_: vec::IntoIter<T>) -> Self {
            panic!()
        }
    }
}

// Site 4: an operator, where the sole specialization is derive-generated (#55243).
mod derived_specializer {
    use std::borrow::Borrow;

    #[derive(PartialEq)]
    struct MyString(String);

    impl Borrow<str> for MyString {
        fn borrow(&self) -> &str {
            &self.0
        }
    }

    impl<Rhs> PartialEq<Rhs> for MyString
    where
        Rhs: ?Sized + Borrow<str>,
    {
        default fn eq(&self, rhs: &Rhs) -> bool {
            self.0 == rhs.borrow()
        }
    }

    fn use_it() {
        // Should select `PartialEq<str>`; the derived `PartialEq<MyString>` is the
        // sole specialization and inference commits `Rhs = MyString`.
        let s = MyString(String::from("Hello, world!"));
        let _ = s == "Hello, world!";
    }
}

fn main() {}
