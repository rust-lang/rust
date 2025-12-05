#![warn(clippy::implicit_clone)]
#![allow(clippy::clone_on_copy, clippy::redundant_clone)]
use std::borrow::Borrow;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

fn return_owned_from_slice(slice: &[u32]) -> Vec<u32> {
    slice.to_owned()
}

pub fn own_same<T>(v: T) -> T
where
    T: ToOwned<Owned = T>,
{
    v.to_owned()
}

pub fn own_same_from_ref<T>(v: &T) -> T
where
    T: ToOwned<Owned = T>,
{
    v.to_owned()
}

pub fn own_different<T, U>(v: T) -> U
where
    T: ToOwned<Owned = U>,
{
    v.to_owned()
}

#[derive(Copy, Clone)]
struct Kitten;
impl Kitten {
    // badly named method
    fn to_vec(self) -> Kitten {
        Kitten {}
    }
}
impl Borrow<BorrowedKitten> for Kitten {
    fn borrow(&self) -> &BorrowedKitten {
        static VALUE: BorrowedKitten = BorrowedKitten {};
        &VALUE
    }
}

struct BorrowedKitten;
impl ToOwned for BorrowedKitten {
    type Owned = Kitten;
    fn to_owned(&self) -> Kitten {
        Kitten {}
    }
}

mod weird {
    #[allow(clippy::ptr_arg)]
    pub fn to_vec(v: &Vec<u32>) -> Vec<u32> {
        v.clone()
    }
}

fn main() {
    let vec = vec![5];
    let _ = return_owned_from_slice(&vec);
    let _ = vec.to_owned();
    //~^ implicit_clone
    let _ = vec.to_vec();
    //~^ implicit_clone

    let vec_ref = &vec;
    let _ = return_owned_from_slice(vec_ref);
    let _ = vec_ref.to_owned();
    let _ = vec_ref.to_vec();
    //~^ implicit_clone

    // we expect no lint for this
    let _ = weird::to_vec(&vec);

    // we expect no lints for this
    let slice: &[u32] = &[1, 2, 3, 4, 5];
    let _ = return_owned_from_slice(slice);
    let _ = slice.to_owned();
    let _ = slice.to_vec();

    let str = "hello world".to_string();
    let _ = str.to_owned();
    //~^ implicit_clone

    // testing w/ an arbitrary type
    let kitten = Kitten {};
    let _ = kitten.to_owned();
    //~^ implicit_clone
    let _ = own_same_from_ref(&kitten);
    // this shouldn't lint
    let _ = kitten.to_vec();

    // we expect no lints for this
    let borrowed = BorrowedKitten {};
    let _ = borrowed.to_owned();

    let pathbuf = PathBuf::new();
    let _ = pathbuf.to_owned();
    //~^ implicit_clone
    let _ = pathbuf.to_path_buf();
    //~^ implicit_clone

    let os_string = OsString::from("foo");
    let _ = os_string.to_owned();
    //~^ implicit_clone
    let _ = os_string.to_os_string();
    //~^ implicit_clone

    // we expect no lints for this
    let os_str = OsStr::new("foo");
    let _ = os_str.to_owned();
    let _ = os_str.to_os_string();

    // issue #8227
    let pathbuf_ref = &pathbuf;
    let pathbuf_ref = &pathbuf_ref;
    let _ = pathbuf_ref.to_owned(); // Don't lint. Returns `&PathBuf`
    let _ = pathbuf_ref.to_path_buf();
    //~^ implicit_clone
    let pathbuf_ref = &pathbuf_ref;
    let _ = pathbuf_ref.to_owned(); // Don't lint. Returns `&&PathBuf`
    let _ = pathbuf_ref.to_path_buf();
    //~^ implicit_clone

    struct NoClone;
    impl ToOwned for NoClone {
        type Owned = Self;
        fn to_owned(&self) -> Self {
            NoClone
        }
    }
    let no_clone = &NoClone;
    let _ = no_clone.to_owned();

    let s = String::from("foo");
    let _ = s.to_owned();
    //~^ implicit_clone
    let _ = s.to_string();
    //~^ implicit_clone
}
