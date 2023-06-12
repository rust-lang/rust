//@run-rustfix
// rustfix-only-machine-applicable
#![feature(lint_reasons)]
#![warn(clippy::redundant_clone)]
#![allow(
    clippy::drop_non_drop,
    clippy::implicit_clone,
    clippy::uninlined_format_args,
    clippy::unnecessary_literal_unwrap
)]

use std::ffi::OsString;
use std::path::Path;

fn main() {
    let _s = ["lorem", "ipsum"].join(" ").to_string();

    let s = String::from("foo");
    let _s = s.clone();

    let s = String::from("foo");
    let _s = s.to_string();

    let s = String::from("foo");
    let _s = s.to_owned();

    let _s = Path::new("/a/b/").join("c").to_owned();

    let _s = Path::new("/a/b/").join("c").to_path_buf();

    let _s = OsString::new().to_owned();

    let _s = OsString::new().to_os_string();

    // Check that lint level works
    #[allow(clippy::redundant_clone)]
    let _s = String::new().to_string();

    // Check that lint level works
    #[expect(clippy::redundant_clone)]
    let _s = String::new().to_string();

    let tup = (String::from("foo"),);
    let _t = tup.0.clone();

    let tup_ref = &(String::from("foo"),);
    let _s = tup_ref.0.clone(); // this `.clone()` cannot be removed

    {
        let x = String::new();
        let y = &x;

        let _x = x.clone(); // ok; `x` is borrowed by `y`

        let _ = y.len();
    }

    let x = (String::new(),);
    let _ = Some(String::new()).unwrap_or_else(|| x.0.clone()); // ok; closure borrows `x`

    with_branch(Alpha, true);
    cannot_double_move(Alpha);
    cannot_move_from_type_with_drop();
    borrower_propagation();
    not_consumed();
    issue_5405();
    manually_drop();
    clone_then_move_cloned();
    hashmap_neg();
    false_negative_5707();
}

#[derive(Clone)]
struct Alpha;
fn with_branch(a: Alpha, b: bool) -> (Alpha, Alpha) {
    if b { (a.clone(), a.clone()) } else { (Alpha, a) }
}

fn cannot_double_move(a: Alpha) -> (Alpha, Alpha) {
    (a.clone(), a)
}

struct TypeWithDrop {
    x: String,
}

impl Drop for TypeWithDrop {
    fn drop(&mut self) {}
}

fn cannot_move_from_type_with_drop() -> String {
    let s = TypeWithDrop { x: String::new() };
    s.x.clone() // removing this `clone()` summons E0509
}

fn borrower_propagation() {
    let s = String::new();
    let t = String::new();

    {
        fn b() -> bool {
            unimplemented!()
        }
        let _u = if b() { &s } else { &t };

        // ok; `s` and `t` are possibly borrowed
        let _s = s.clone();
        let _t = t.clone();
    }

    {
        let _u = || s.len();
        let _v = [&t; 32];
        let _s = s.clone(); // ok
        let _t = t.clone(); // ok
    }

    {
        let _u = {
            let u = Some(&s);
            let _ = s.clone(); // ok
            u
        };
        let _s = s.clone(); // ok
    }

    {
        use std::convert::identity as id;
        let _u = id(id(&s));
        let _s = s.clone(); // ok, `u` borrows `s`
    }

    let _s = s.clone();
    let _t = t.clone();

    #[derive(Clone)]
    struct Foo {
        x: usize,
    }

    {
        let f = Foo { x: 123 };
        let _x = Some(f.x);
        let _f = f.clone();
    }

    {
        let f = Foo { x: 123 };
        let _x = &f.x;
        let _f = f.clone(); // ok
    }
}

fn not_consumed() {
    let x = std::path::PathBuf::from("home");
    let y = x.clone().join("matthias");
    // join() creates a new owned PathBuf, does not take a &mut to x variable, thus the .clone() is
    // redundant. (It also does not consume the PathBuf)

    println!("x: {:?}, y: {:?}", x, y);

    let mut s = String::new();
    s.clone().push_str("foo"); // OK, removing this `clone()` will change the behavior.
    s.push_str("bar");
    assert_eq!(s, "bar");

    let t = Some(s);
    // OK
    if let Some(x) = t.clone() {
        println!("{}", x);
    }
    if let Some(x) = t {
        println!("{}", x);
    }
}

#[allow(clippy::clone_on_copy)]
fn issue_5405() {
    let a: [String; 1] = [String::from("foo")];
    let _b: String = a[0].clone();

    let c: [usize; 2] = [2, 3];
    let _d: usize = c[1].clone();
}

fn manually_drop() {
    use std::mem::ManuallyDrop;
    use std::sync::Arc;

    let a = ManuallyDrop::new(Arc::new("Hello!".to_owned()));
    let _ = a.clone(); // OK

    let p: *const String = Arc::into_raw(ManuallyDrop::into_inner(a));
    unsafe {
        Arc::from_raw(p);
        Arc::from_raw(p);
    }
}

fn clone_then_move_cloned() {
    // issue #5973
    let x = Some(String::new());
    // ok, x is moved while the clone is in use.
    assert_eq!(x.clone(), None, "not equal {}", x.unwrap());

    // issue #5595
    fn foo<F: Fn()>(_: &Alpha, _: F) {}
    let x = Alpha;
    // ok, data is moved while the clone is in use.
    foo(&x.clone(), move || {
        let _ = x;
    });

    // issue #6998
    struct S(String);
    impl S {
        fn m(&mut self) {}
    }
    let mut x = S(String::new());
    x.0.clone().chars().for_each(|_| x.m());
}

fn hashmap_neg() {
    // issue 5707
    use std::collections::HashMap;
    use std::path::PathBuf;

    let p = PathBuf::from("/");

    let mut h: HashMap<&str, &str> = HashMap::new();
    h.insert("orig-p", p.to_str().unwrap());

    let mut q = p.clone();
    q.push("foo");

    println!("{:?} {}", h, q.display());
}

fn false_negative_5707() {
    fn foo(_x: &Alpha, _y: &mut Alpha) {}

    let x = Alpha;
    let mut y = Alpha;
    foo(&x, &mut y);
    let _z = x.clone(); // pr 7346 can't lint on `x`
    drop(y);
}
