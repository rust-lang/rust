#![deny(clippy::useless_asref)]
#![allow(
    clippy::explicit_auto_deref,
    clippy::uninlined_format_args,
    clippy::map_clone,
    clippy::needless_pass_by_ref_mut,
    clippy::redundant_closure
)]

use std::fmt::Debug;
use std::ops::Deref;
use std::rc::{Rc, Weak as RcWeak};
use std::sync::{Arc, Weak as ArcWeak};

struct FakeAsRef;

#[allow(clippy::should_implement_trait)]
impl FakeAsRef {
    fn as_ref(&self) -> &Self {
        self
    }
}

struct MoreRef;

impl<'a, 'b, 'c> AsRef<&'a &'b &'c MoreRef> for MoreRef {
    fn as_ref(&self) -> &&'a &'b &'c MoreRef {
        &&&&MoreRef
    }
}

fn foo_rstr(x: &str) {
    println!("{:?}", x);
}
fn foo_rslice(x: &[i32]) {
    println!("{:?}", x);
}
fn foo_mrslice(x: &mut [i32]) {
    println!("{:?}", x);
}
fn foo_rrrrmr(_: &&&&MoreRef) {
    println!("so many refs");
}

fn not_ok() {
    let rstr: &str = "hello";
    let mut mrslice: &mut [i32] = &mut [1, 2, 3];

    {
        let rslice: &[i32] = &*mrslice;
        foo_rstr(rstr.as_ref());
        //~^ useless_asref
        foo_rstr(rstr);
        foo_rslice(rslice.as_ref());
        //~^ useless_asref
        foo_rslice(rslice);
    }
    {
        foo_mrslice(mrslice.as_mut());
        //~^ useless_asref
        foo_mrslice(mrslice);
        foo_rslice(mrslice.as_ref());
        //~^ useless_asref
        foo_rslice(mrslice);
    }

    {
        let rrrrrstr = &&&&rstr;
        let rrrrrslice = &&&&&*mrslice;
        foo_rslice(rrrrrslice.as_ref());
        //~^ useless_asref
        foo_rslice(rrrrrslice);
        foo_rstr(rrrrrstr.as_ref());
        //~^ useless_asref
        foo_rstr(rrrrrstr);
    }
    {
        let mrrrrrslice = &mut &mut &mut &mut mrslice;
        foo_mrslice(mrrrrrslice.as_mut());
        //~^ useless_asref
        foo_mrslice(mrrrrrslice);
        foo_rslice(mrrrrrslice.as_ref());
        //~^ useless_asref
        foo_rslice(mrrrrrslice);
    }
    #[allow(unused_parens, clippy::double_parens, clippy::needless_borrow)]
    foo_rrrrmr((&&&&MoreRef).as_ref());
    //~^ useless_asref

    generic_not_ok(mrslice);
    generic_ok(mrslice);
}

fn ok() {
    let string = "hello".to_owned();
    let mut arr = [1, 2, 3];
    let mut vec = vec![1, 2, 3];

    {
        foo_rstr(string.as_ref());
        foo_rslice(arr.as_ref());
        foo_rslice(vec.as_ref());
    }
    {
        foo_mrslice(arr.as_mut());
        foo_mrslice(vec.as_mut());
    }

    {
        let rrrrstring = &&&&string;
        let rrrrarr = &&&&arr;
        let rrrrvec = &&&&vec;
        foo_rstr(rrrrstring.as_ref());
        foo_rslice(rrrrarr.as_ref());
        foo_rslice(rrrrvec.as_ref());
    }
    {
        let mrrrrarr = &mut &mut &mut &mut arr;
        let mrrrrvec = &mut &mut &mut &mut vec;
        foo_mrslice(mrrrrarr.as_mut());
        foo_mrslice(mrrrrvec.as_mut());
    }
    FakeAsRef.as_ref();
    foo_rrrrmr(MoreRef.as_ref());

    generic_not_ok(arr.as_mut());
    generic_ok(&mut arr);
}

fn foo_mrt<T: Debug + ?Sized>(t: &mut T) {
    println!("{:?}", t);
}
fn foo_rt<T: Debug + ?Sized>(t: &T) {
    println!("{:?}", t);
}

fn generic_not_ok<T: AsMut<T> + AsRef<T> + Debug + ?Sized>(mrt: &mut T) {
    foo_mrt(mrt.as_mut());
    //~^ useless_asref
    foo_mrt(mrt);
    foo_rt(mrt.as_ref());
    //~^ useless_asref
    foo_rt(mrt);
}

fn generic_ok<U: AsMut<T> + AsRef<T> + ?Sized, T: Debug + ?Sized>(mru: &mut U) {
    foo_mrt(mru.as_mut());
    foo_rt(mru.as_ref());
}

fn foo() {
    let x = Some(String::new());
    let z = x.as_ref().map(String::clone);
    //~^ useless_asref

    let z = x.as_ref().map(|z| z.clone());
    //~^ useless_asref

    let z = x.as_ref().map(|z| String::clone(z));
    //~^ useless_asref
}

mod issue12135 {
    pub struct Struct {
        field: Option<InnerStruct>,
    }

    #[derive(Clone)]
    pub struct Foo;

    #[derive(Clone)]
    struct InnerStruct {
        x: Foo,
    }

    impl InnerStruct {
        fn method(&self) -> &Foo {
            &self.x
        }
    }

    pub fn f(x: &Struct) -> Option<Foo> {
        x.field.as_ref().map(|v| v.clone());
        //~^ useless_asref

        x.field.as_ref().map(Clone::clone);
        //~^ useless_asref

        x.field.as_ref().map(|v| Clone::clone(v));
        //~^ useless_asref

        // https://github.com/rust-lang/rust-clippy/pull/12136#discussion_r1451565223
        #[allow(clippy::clone_on_copy)]
        Some(1).as_ref().map(|&x| x.clone());
        //~^ useless_asref

        x.field.as_ref().map(|v| v.method().clone())
    }
}

fn issue_12528() {
    struct Foo;

    let opt = Some(Arc::new(Foo));
    let _ = opt.as_ref().map(Arc::clone);

    let opt = Some(Rc::new(Foo));
    let _ = opt.as_ref().map(Rc::clone);

    let opt = Some(Arc::downgrade(&Arc::new(Foo)));
    let _ = opt.as_ref().map(ArcWeak::clone);

    let opt = Some(Rc::downgrade(&Rc::new(Foo)));
    let _ = opt.as_ref().map(RcWeak::clone);
}

fn issue_14088() {
    let s = Some("foo");
    let _: Option<&str> = s.as_ref().map(|x| x.as_ref());
}

pub struct Wrap<T> {
    inner: T,
}

impl<T> Deref for Wrap<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

struct NonCloneableError;

pub struct Issue12357 {
    current: Option<Wrap<Arc<u32>>>,
}

impl Issue12357 {
    fn f(&self) -> Option<Arc<u32>> {
        self.current.as_ref().map(|p| Arc::clone(p))
    }

    fn g(&self) {
        let result: Result<String, NonCloneableError> = Ok("Hello".to_string());
        let cloned = result.as_ref().map(|s| s.clone());
    }
}

fn issue_14828() {
    pub trait T {
        fn as_ref(&self) {}
    }

    impl T for () {}

    ().as_ref();
}

fn main() {
    not_ok();
    ok();
}
