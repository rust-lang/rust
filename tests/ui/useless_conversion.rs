#![deny(clippy::useless_conversion)]
#![allow(clippy::needless_if, clippy::unnecessary_wraps)]

fn test_generic<T: Copy>(val: T) -> T {
    let _ = T::from(val);
    val.into()
}

fn test_generic2<T: Copy + Into<i32> + Into<U>, U: From<T>>(val: T) {
    // ok
    let _: i32 = val.into();
    let _: U = val.into();
    let _ = U::from(val);
}

fn test_questionmark() -> Result<(), ()> {
    {
        let _: i32 = 0i32.into();
        Ok(Ok(()))
    }??;
    Ok(())
}

fn test_issue_3913() -> Result<(), std::io::Error> {
    use std::fs;
    use std::path::Path;

    let path = Path::new(".");
    for _ in fs::read_dir(path)? {}

    Ok(())
}

fn dont_lint_on_type_alias() {
    type A = i32;
    _ = A::from(0i32);
}

fn dont_lint_into_iter_on_immutable_local_implementing_iterator_in_expr() {
    let text = "foo\r\nbar\n\nbaz\n";
    let lines = text.lines();
    if Some("ok") == lines.into_iter().next() {}
}

fn lint_into_iter_on_mutable_local_implementing_iterator_in_expr() {
    let text = "foo\r\nbar\n\nbaz\n";
    let mut lines = text.lines();
    if Some("ok") == lines.into_iter().next() {}
}

fn lint_into_iter_on_expr_implementing_iterator() {
    let text = "foo\r\nbar\n\nbaz\n";
    let mut lines = text.lines().into_iter();
    if Some("ok") == lines.next() {}
}

fn lint_into_iter_on_expr_implementing_iterator_2() {
    let text = "foo\r\nbar\n\nbaz\n";
    if Some("ok") == text.lines().into_iter().next() {}
}

#[allow(const_item_mutation)]
fn lint_into_iter_on_const_implementing_iterator() {
    const NUMBERS: std::ops::Range<i32> = 0..10;
    let _ = NUMBERS.into_iter().next();
}

fn lint_into_iter_on_const_implementing_iterator_2() {
    const NUMBERS: std::ops::Range<i32> = 0..10;
    let mut n = NUMBERS.into_iter();
    n.next();
}

#[derive(Clone, Copy)]
struct CopiableCounter {
    counter: u32,
}

impl Iterator for CopiableCounter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.counter = self.counter.wrapping_add(1);
        Some(self.counter)
    }
}

fn dont_lint_into_iter_on_copy_iter() {
    let mut c = CopiableCounter { counter: 0 };
    assert_eq!(c.into_iter().next(), Some(1));
    assert_eq!(c.into_iter().next(), Some(1));
    assert_eq!(c.next(), Some(1));
    assert_eq!(c.next(), Some(2));
}

fn dont_lint_into_iter_on_static_copy_iter() {
    static mut C: CopiableCounter = CopiableCounter { counter: 0 };
    unsafe {
        assert_eq!(C.into_iter().next(), Some(1));
        assert_eq!(C.into_iter().next(), Some(1));
        assert_eq!(C.next(), Some(1));
        assert_eq!(C.next(), Some(2));
    }
}

fn main() {
    test_generic(10i32);
    test_generic2::<i32, i32>(10i32);
    test_questionmark().unwrap();
    test_issue_3913().unwrap();

    dont_lint_on_type_alias();
    dont_lint_into_iter_on_immutable_local_implementing_iterator_in_expr();
    lint_into_iter_on_mutable_local_implementing_iterator_in_expr();
    lint_into_iter_on_expr_implementing_iterator();
    lint_into_iter_on_expr_implementing_iterator_2();
    lint_into_iter_on_const_implementing_iterator();
    lint_into_iter_on_const_implementing_iterator_2();
    dont_lint_into_iter_on_copy_iter();
    dont_lint_into_iter_on_static_copy_iter();

    let _: String = "foo".into();
    let _: String = From::from("foo");
    let _ = String::from("foo");
    #[allow(clippy::useless_conversion)]
    {
        let _: String = "foo".into();
        let _ = String::from("foo");
        let _ = "".lines().into_iter();
    }

    let _: String = "foo".to_string().into();
    let _: String = From::from("foo".to_string());
    let _ = String::from("foo".to_string());
    let _ = String::from(format!("A: {:04}", 123));
    let _ = "".lines().into_iter();
    let _ = vec![1, 2, 3].into_iter().into_iter();
    let _: String = format!("Hello {}", "world").into();

    // keep parentheses around `a + b` for suggestion (see #4750)
    let a: i32 = 1;
    let b: i32 = 1;
    let _ = i32::from(a + b) * 3;

    // see #7205
    let s: Foo<'a'> = Foo;
    let _: Foo<'b'> = s.into();
    let s2: Foo<'a'> = Foo;
    let _: Foo<'a'> = s2.into();
    let s3: Foo<'a'> = Foo;
    let _ = Foo::<'a'>::from(s3);
    let s4: Foo<'a'> = Foo;
    let _ = vec![s4, s4, s4].into_iter().into_iter();

    issue11300::bar();
}

#[allow(dead_code)]
fn issue11065_fp() {
    use std::option::IntoIter;
    fn takes_into_iter(_: impl IntoIterator<Item = i32>) {}

    macro_rules! x {
        ($e:expr) => {
            takes_into_iter($e);
            let _: IntoIter<i32> = $e; // removing `.into_iter()` leads to a type error here
        };
    }
    x!(Some(5).into_iter());
}

#[allow(dead_code)]
fn explicit_into_iter_fn_arg() {
    fn a<T>(_: T) {}
    fn b<T: IntoIterator<Item = i32>>(_: T) {}
    fn c(_: impl IntoIterator<Item = i32>) {}
    fn d<T>(_: T)
    where
        T: IntoIterator<Item = i32>,
    {
    }
    fn f(_: std::vec::IntoIter<i32>) {}

    a(vec![1, 2].into_iter());
    b(vec![1, 2].into_iter());
    c(vec![1, 2].into_iter());
    d(vec![1, 2].into_iter());
    b([&1, &2, &3].into_iter().cloned());

    b(vec![1, 2].into_iter().into_iter());
    b(vec![1, 2].into_iter().into_iter().into_iter());

    macro_rules! macro_generated {
        () => {
            vec![1, 2].into_iter()
        };
    }
    b(macro_generated!());
}

mod issue11300 {
    pub fn foo<I>(i: I)
    where
        I: IntoIterator<Item = i32> + ExactSizeIterator,
    {
        assert_eq!(i.len(), 3);
    }

    trait Helper<T: ?Sized> {}
    impl Helper<i32> for [i32; 3] {}
    impl Helper<i32> for std::array::IntoIter<i32, 3> {}
    impl Helper<()> for std::array::IntoIter<i32, 3> {}

    fn foo2<X: ?Sized, I>(_: I)
    where
        I: IntoIterator<Item = i32> + Helper<X>,
    {
    }

    trait Helper2<T> {}
    impl Helper2<std::array::IntoIter<i32, 3>> for i32 {}
    impl Helper2<[i32; 3]> for i32 {}
    fn foo3<I>(_: I)
    where
        I: IntoIterator<Item = i32>,
        i32: Helper2<I>,
    {
    }

    pub fn bar() {
        // This should not trigger the lint:
        // `[i32, 3]` does not satisfy the `ExactSizeIterator` bound, so the into_iter call cannot be
        // removed and is not useless.
        foo([1, 2, 3].into_iter());

        // This should trigger the lint, receiver type [i32; 3] also implements `Helper`
        foo2::<i32, _>([1, 2, 3].into_iter());

        // This again should *not* lint, since X = () and I = std::array::IntoIter<i32, 3>,
        // and `[i32; 3]: Helper<()>` is not true (only `std::array::IntoIter<i32, 3>: Helper<()>` is).
        foo2::<(), _>([1, 2, 3].into_iter());

        // This should lint. Removing the `.into_iter()` means that `I` gets substituted with `[i32; 3]`,
        // and `i32: Helper2<[i32, 3]>` is true, so this call is indeed unncessary.
        foo3([1, 2, 3].into_iter());
    }

    fn ice() {
        struct S1;
        impl S1 {
            pub fn foo<I: IntoIterator>(&self, _: I) {}
        }

        S1.foo([1, 2].into_iter());

        // ICE that occured in itertools
        trait Itertools {
            fn interleave_shortest<J>(self, other: J)
            where
                J: IntoIterator,
                Self: Sized;
        }
        impl<I: Iterator> Itertools for I {
            fn interleave_shortest<J>(self, other: J)
            where
                J: IntoIterator,
                Self: Sized,
            {
            }
        }
        let v0: Vec<i32> = vec![0, 2, 4];
        let v1: Vec<i32> = vec![1, 3, 5, 7];
        v0.into_iter().interleave_shortest(v1.into_iter());

        trait TraitWithLifetime<'a> {}
        impl<'a> TraitWithLifetime<'a> for std::array::IntoIter<&'a i32, 2> {}

        struct Helper;
        impl<'a> Helper {
            fn with_lt<I>(&self, _: I)
            where
                I: IntoIterator + TraitWithLifetime<'a>,
            {
            }
        }
        Helper.with_lt([&1, &2].into_iter());
    }
}

#[derive(Copy, Clone)]
struct Foo<const C: char>;

impl From<Foo<'a'>> for Foo<'b'> {
    fn from(_s: Foo<'a'>) -> Self {
        Foo
    }
}
