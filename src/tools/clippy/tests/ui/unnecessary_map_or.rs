//@aux-build:proc_macros.rs
#![warn(clippy::unnecessary_map_or)]
#![allow(clippy::no_effect)]
#![allow(clippy::eq_op)]
#![allow(clippy::unnecessary_lazy_evaluations)]
#![allow(clippy::nonminimal_bool)]
#[clippy::msrv = "1.70.0"]
#[macro_use]
extern crate proc_macros;

fn main() {
    // should trigger
    let _ = Some(5).map_or(false, |n| n == 5);
    //~^ unnecessary_map_or
    let _ = Some(5).map_or(true, |n| n != 5);
    //~^ unnecessary_map_or
    let _ = Some(5).map_or(false, |n| {
        //~^ unnecessary_map_or
        let _ = 1;
        n == 5
    });
    let _ = Some(5).map_or(false, |n| {
        //~^ unnecessary_map_or
        let _ = n;
        6 >= 5
    });
    let _ = Some(vec![5]).map_or(false, |n| n == [5]);
    //~^ unnecessary_map_or
    let _ = Some(vec![1]).map_or(false, |n| vec![2] == n);
    //~^ unnecessary_map_or
    let _ = Some(5).map_or(false, |n| n == n);
    //~^ unnecessary_map_or
    let _ = Some(5).map_or(false, |n| n == if 2 > 1 { n } else { 0 });
    //~^ unnecessary_map_or
    let _ = Ok::<Vec<i32>, i32>(vec![5]).map_or(false, |n| n == [5]);
    //~^ unnecessary_map_or
    let _ = Ok::<i32, i32>(5).map_or(false, |n| n == 5);
    //~^ unnecessary_map_or
    let _ = Some(5).map_or(false, |n| n == 5).then(|| 1);
    //~^ unnecessary_map_or
    let _ = Some(5).map_or(true, |n| n == 5);
    //~^ unnecessary_map_or
    let _ = Some(5).map_or(true, |n| 5 == n);
    //~^ unnecessary_map_or
    let _ = !Some(5).map_or(false, |n| n == 5);
    //~^ unnecessary_map_or
    let _ = Some(5).map_or(false, |n| n == 5) || false;
    //~^ unnecessary_map_or
    let _ = Some(5).map_or(false, |n| n == 5) as usize;
    //~^ unnecessary_map_or

    macro_rules! x {
        () => {
            Some(1)
        };
    }
    // methods lints dont fire on macros
    let _ = x!().map_or(false, |n| n == 1);
    let _ = x!().map_or(false, |n| n == vec![1][0]);

    msrv_1_69();

    external! {
        let _ = Some(5).map_or(false, |n| n == 5);
    }

    with_span! {
        let _ = Some(5).map_or(false, |n| n == 5);
    }

    // check for presence of PartialEq, and alter suggestion to use `is_ok_and` if absent
    struct S;
    let r: Result<i32, S> = Ok(3);
    let _ = r.map_or(false, |x| x == 7);
    //~^ unnecessary_map_or

    // lint constructs that are not comparaisons as well
    let func = |_x| true;
    let r: Result<i32, S> = Ok(3);
    let _ = r.map_or(false, func);
    //~^ unnecessary_map_or
    let _ = Some(5).map_or(false, func);
    //~^ unnecessary_map_or
    let _ = Some(5).map_or(true, func);
    //~^ unnecessary_map_or

    #[derive(PartialEq)]
    struct S2;
    let r: Result<i32, S2> = Ok(4);
    let _ = r.map_or(false, |x| x == 8);
    //~^ unnecessary_map_or

    // do not lint `Result::map_or(true, …)`
    let r: Result<i32, S2> = Ok(4);
    let _ = r.map_or(true, |x| x == 8);
}

#[clippy::msrv = "1.69.0"]
fn msrv_1_69() {
    // is_some_and added in 1.70.0
    let _ = Some(5).map_or(false, |n| n == if 2 > 1 { n } else { 0 });
}

#[clippy::msrv = "1.81.0"]
fn msrv_1_81() {
    // is_none_or added in 1.82.0
    let _ = Some(5).map_or(true, |n| n == if 2 > 1 { n } else { 0 });
}

fn with_refs(o: &mut Option<u32>) -> bool {
    o.map_or(true, |n| n > 5) || (o as &Option<u32>).map_or(true, |n| n < 5)
    //~^ unnecessary_map_or
    //~| unnecessary_map_or
}

struct S;

impl std::ops::Deref for S {
    type Target = Option<u32>;
    fn deref(&self) -> &Self::Target {
        &Some(0)
    }
}

fn with_deref(o: &S) -> bool {
    o.map_or(true, |n| n > 5)
    //~^ unnecessary_map_or
}

fn issue14201(a: Option<String>, b: Option<String>, s: &String) -> bool {
    let x = a.map_or(false, |a| a == *s);
    //~^ unnecessary_map_or
    let y = b.map_or(true, |b| b == *s);
    //~^ unnecessary_map_or
    x && y
}

fn issue14714() {
    assert!(Some("test").map_or(false, |x| x == "test"));
    //~^ unnecessary_map_or

    // even though we're in a macro context, we still need to parenthesise because of the `then`
    assert!(Some("test").map_or(false, |x| x == "test").then(|| 1).is_some());
    //~^ unnecessary_map_or

    // method lints don't fire on macros
    macro_rules! m {
        ($x:expr) => {
            // should become !($x == Some(1))
            let _ = !$x.map_or(false, |v| v == 1);
            // should become $x == Some(1)
            let _ = $x.map_or(false, |v| v == 1);
        };
    }
    m!(Some(5));
}

fn issue15180() {
    let s = std::sync::Mutex::new(Some("foo"));
    _ = s.lock().unwrap().map_or(false, |s| s == "foo");
    //~^ unnecessary_map_or

    let s = &&&&Some("foo");
    _ = s.map_or(false, |s| s == "foo");
    //~^ unnecessary_map_or
}
