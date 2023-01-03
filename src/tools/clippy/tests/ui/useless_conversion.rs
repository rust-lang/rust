// run-rustfix

#![deny(clippy::useless_conversion)]
#![allow(clippy::unnecessary_wraps)]

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
}

#[derive(Copy, Clone)]
struct Foo<const C: char>;

impl From<Foo<'a'>> for Foo<'b'> {
    fn from(_s: Foo<'a'>) -> Self {
        Foo
    }
}
