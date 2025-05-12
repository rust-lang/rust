//@ run-pass
#![allow(unused_macro_rules)]

//{{{ issue 40569 ==============================================================

macro_rules! my_struct {
    ($(#[$meta:meta])* $ident:ident) => {
        $(#[$meta])* struct $ident;
    }
}

my_struct!(#[derive(Debug, PartialEq)] Foo40569);

fn test_40569() {
    assert_eq!(Foo40569, Foo40569);
}

//}}}

//{{{ issue 26444 ==============================================================

macro_rules! foo_26444 {
    ($($beginning:ident),*; $middle:ident; $($end:ident),*) => {
        stringify!($($beginning,)* $middle $(,$end)*)
    }
}

fn test_26444() {
    assert_eq!("a, b, c, d, e", foo_26444!(a, b; c; d, e));
    assert_eq!("f", foo_26444!(; f ;));
}

macro_rules! pat_26444 {
    ($fname:ident $($arg:pat)* =) => {}
}

pat_26444!(foo 1 2 5...7 =);
pat_26444!(bar Some(ref x) Ok(ref mut y) &(w, z) =);

//}}}

//{{{ issue 40984 ==============================================================

macro_rules! thread_local_40984 {
    () => {};
    ($(#[$attr:meta])* $vis:vis static $name:ident: $t:ty = $init:expr; $($rest:tt)*) => {
        thread_local_40984!($($rest)*);
    };
    ($(#[$attr:meta])* $vis:vis static $name:ident: $t:ty = $init:expr) => {};
}

thread_local_40984! {
    // no docs
    #[allow(unused)]
    static FOO: i32 = 42;
    /// docs
    pub static BAR: String = String::from("bar");

    // look at these restrictions!!
    pub(crate) static BAZ: usize = 0;
    pub(in foo) static QUUX: usize = 0;
}

//}}}

//{{{ issue 35650 ==============================================================

macro_rules! size {
    ($ty:ty) => {
        std::mem::size_of::<$ty>()
    };
    ($size:tt) => {
        $size
    };
}

fn test_35650() {
    assert_eq!(size!(u64), 8);
    assert_eq!(size!(5), 5);
}

//}}}

//{{{ issue 27832 ==============================================================

macro_rules! m {
    ( $i:ident ) => ();
    ( $t:tt $j:tt ) => ();
}

m!(c);
m!(t 9);
m!(0 9);
m!(struct);
m!(struct Foo);

macro_rules! m2 {
    ( $b:expr ) => ();
    ( $t:tt $u:tt ) => ();
}

m2!(3);
m2!(1 2);
m2!(_ 1);
m2!(enum Foo);

//}}}

//{{{ issue 39964 ==============================================================

macro_rules! foo_39964 {
    ($a:ident) => {};
    (_) => {};
}

foo_39964!(_);

//}}}

//{{{ issue 34030 ==============================================================

macro_rules! foo_34030 {
    ($($t:ident),* /) => {};
}

foo_34030!(a, b/);
foo_34030!(a/);
foo_34030!(/);

//}}}

//{{{ issue 24189 ==============================================================

macro_rules! foo_24189 {
    (
        pub enum $name:ident {
            $( #[$attr:meta] )* $var:ident
        }
    ) => {
        pub enum $name {
            $( #[$attr] )* $var
        }
    };
}

foo_24189! {
    pub enum Foo24189 {
        #[doc = "Bar"] Baz
    }
}

macro_rules! serializable {
    (
        $(#[$struct_meta:meta])*
        pub struct $name:ident {
            $(
                $(#[$field_meta:meta])*
                $field:ident: $type_:ty
            ),* ,
        }
    ) => {
        $(#[$struct_meta])*
        pub struct $name {
            $(
                $(#[$field_meta])*
                $field: $type_
            ),* ,
        }
    }
}

serializable! {
    #[allow(dead_code)]
    /// This is a test
    pub struct Tester {
        #[allow(dead_code)]
        name: String,
    }
}

macro_rules! foo_24189_c {
    ( $( > )* $x:ident ) => { };
}
foo_24189_c!( > a );

fn test_24189() {
    let _ = Foo24189::Baz;
    let _ = Tester { name: "".to_owned() };
}

//}}}

//{{{ issue 50903 ==============================================================

macro_rules! foo_50903 {
    ($($lif:lifetime ,)* #) => {};
}

foo_50903!('a, 'b, #);
foo_50903!('a, #);
foo_50903!(#);

//}}}

//{{{ issue 51477 ==============================================================

macro_rules! foo_51477 {
    ($lifetime:lifetime) => {
        "last token is lifetime"
    };
    ($other:tt) => {
        "last token is other"
    };
    ($first:tt $($rest:tt)*) => {
        foo_51477!($($rest)*)
    };
}

fn test_51477() {
    assert_eq!("last token is lifetime", foo_51477!('a));
    assert_eq!("last token is other", foo_51477!(@));
    assert_eq!("last token is lifetime", foo_51477!(@ {} 'a));
}

//}}}

//{{{ some more tests ==========================================================

macro_rules! test_block {
    (< $($b:block)* >) => {}
}

test_block!(<>);
test_block!(<{}>);
test_block!(<{1}{2}>);

macro_rules! test_ty {
    ($($t:ty),* $(,)*) => {}
}

test_ty!();
test_ty!(,);
test_ty!(u8);
test_ty!(u8,);

macro_rules! test_path {
    ($($t:path),* $(,)*) => {}
}

test_path!();
test_path!(,);
test_path!(::std);
test_path!(std::ops,);
test_path!(any, super, super::super::self::path, X<Y>::Z<'a, T=U>);

macro_rules! test_lifetime {
    (1. $($l:lifetime)* $($b:block)*) => {};
    (2. $($b:block)* $($l:lifetime)*) => {};
}

test_lifetime!(1. 'a 'b {} {});
test_lifetime!(2. {} {} 'a 'b);

//}}}

fn main() {
    test_26444();
    test_40569();
    test_35650();
    test_24189();
    test_51477();
}
