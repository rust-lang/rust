#![feature(box_syntax)]

// Iota-reduction is a rule in the Calculus of (Co-)Inductive Constructions,
// which "says that a destructor applied to an object built from a constructor
// behaves as expected".  -- http://coq.inria.fr/doc/Reference-Manual006.html
//
// It's a little more complicated here, because of pointers and regions and
// trying to get assert failure messages that at least identify which case
// failed.

enum E<T> { Thing(isize, T), Nothing((), ((), ()), [i8; 0]) }
impl<T> E<T> {
    fn is_none(&self) -> bool {
        match *self {
            E::Thing(..) => false,
            E::Nothing(..) => true
        }
    }
    fn get_ref(&self) -> (isize, &T) {
        match *self {
            E::Nothing(..) => panic!("E::get_ref(Nothing::<{}>)",  stringify!(T)),
            E::Thing(x, ref y) => (x, y)
        }
    }
}

macro_rules! check_option {
    ($e:expr, $T:ty) => {{
        check_option!($e, $T, |ptr| assert_eq!(*ptr, $e));
    }};
    ($e:expr, $T:ty, |$v:ident| $chk:expr) => {{
        assert!(None::<$T>.is_none());
        let e = $e;
        let s_ = Some::<$T>(e);
        let $v = s_.as_ref().unwrap();
        $chk
    }}
}

macro_rules! check_fancy {
    ($e:expr, $T:ty) => {{
        check_fancy!($e, $T, |ptr| assert_eq!(*ptr, $e));
    }};
    ($e:expr, $T:ty, |$v:ident| $chk:expr) => {{
        assert!(E::Nothing::<$T>((), ((), ()), [23; 0]).is_none());
        let e = $e;
        let t_ = E::Thing::<$T>(23, e);
        match t_.get_ref() {
            (23, $v) => { $chk }
            _ => panic!("Thing::<{}>(23, {}).get_ref() != (23, _)",
                       stringify!($T), stringify!($e))
        }
    }}
}

macro_rules! check_type {
    ($($a:tt)*) => {{
        check_option!($($a)*);
        check_fancy!($($a)*);
    }}
}

pub fn main() {
    check_type!(&17, &isize);
    check_type!(box 18, Box<isize>);
    check_type!("foo".to_string(), String);
    check_type!(vec![20, 22], Vec<isize>);
    check_type!(main, fn(), |pthing| {
        assert_eq!(main as fn(), *pthing as fn())
    });
}
