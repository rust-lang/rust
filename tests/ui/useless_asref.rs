#![deny(useless_asref)]

struct FakeAsRef;

#[allow(should_implement_trait)]
impl FakeAsRef {
    fn as_ref(&self) -> &Self { self }
}

struct MoreRef;

impl<'a, 'b, 'c> AsRef<&'a &'b &'c MoreRef> for MoreRef {
    fn as_ref(&self) -> &&'a &'b &'c MoreRef {
        &&&&MoreRef
    }
}

fn foo_rstr(x: &str) { println!("{:?}", x); }
fn foo_rslice(x: &[i32]) { println!("{:?}", x); }
fn foo_mrslice(x: &mut [i32]) { println!("{:?}", x); }
fn foo_rrrrmr(_: &&&&MoreRef) { println!("so many refs"); }

fn not_ok() {
    let rstr: &str = "hello";
    let mut mrslice: &mut [i32] = &mut [1,2,3];

    {
        let rslice: &[i32] = &*mrslice;
        foo_rstr(rstr.as_ref());
        foo_rstr(rstr);
        foo_rslice(rslice.as_ref());
        foo_rslice(rslice);
    }
    {
        foo_mrslice(mrslice.as_mut());
        foo_mrslice(mrslice);
        foo_rslice(mrslice.as_ref());
        foo_rslice(mrslice);
    }

    {
        let rrrrrstr = &&&&rstr;
        let rrrrrslice = &&&&&*mrslice;
        foo_rslice(rrrrrslice.as_ref());
        foo_rslice(rrrrrslice);
        foo_rstr(rrrrrstr.as_ref());
        foo_rstr(rrrrrstr);
    }
    {
        let mrrrrrslice = &mut &mut &mut &mut mrslice;
        foo_mrslice(mrrrrrslice.as_mut());
        foo_mrslice(mrrrrrslice);
        foo_rslice(mrrrrrslice.as_ref());
        foo_rslice(mrrrrrslice);
    }
    foo_rrrrmr((&&&&MoreRef).as_ref());
}

fn ok() {
    let string = "hello".to_owned();
    let mut arr = [1,2,3];
    let mut vec = vec![1,2,3];

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
}
fn main() {
    not_ok();
    ok();
}
