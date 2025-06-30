// This is a collection of regression tests related to the NLL problem case 3 that was deferred from
// the implementation of the NLL RFC, and left to be implemented by polonius. Most of them are from
// open issues, e.g. tagged fixed-by-polonius, to ensure that the polonius alpha analysis does
// handle them, as does the datalog implementation.

//@ check-pass
//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: polonius legacy
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] compile-flags: -Z polonius=legacy

use std::collections::HashMap;
use std::hash::Hash;

fn from_the_rfc<'r, K: Hash + Eq + Copy, V: Default>(
    map: &'r mut HashMap<K, V>,
    key: K,
) -> &'r mut V {
    match map.get_mut(&key) {
        Some(value) => value,
        None => {
            map.insert(key, V::default());
            map.get_mut(&key).unwrap()
        }
    }
}

fn issue_112087<'a>(opt: &'a mut Option<i32>, b: bool) -> Result<&'a mut Option<i32>, &'a mut i32> {
    if let Some(v) = opt {
        if b {
            return Err(v);
        }
    }

    *opt = None;
    return Ok(opt);
}

// issue 54663
fn foo(x: &mut u8) -> Option<&u8> {
    if let Some(y) = bar(x) {
        return Some(y);
    }
    bar(x)
}

fn bar(x: &mut u8) -> Option<&u8> {
    Some(x)
}

// issue 123839
struct Foo {
    val: i32,
    status: i32,
    err_str: String,
}

impl Foo {
    fn bar(self: &mut Self) -> Result<(), &str> {
        if self.val == 0 {
            self.status = -1;
            Err("val is zero")
        } else if self.val < 0 {
            self.status = -2;
            self.err_str = format!("unexpected negative val {}", self.val);
            Err(&self.err_str)
        } else {
            Ok(())
        }
    }
    fn foo(self: &mut Self) -> Result<(), &str> {
        self.bar()?; // rust reports this line conflicts with the next line
        self.status = 1; // and this line is the victim
        Ok(())
    }
}

// issue 124070
struct S {
    field: String,
}

impl S {
    fn f(&mut self) -> &str {
        let a = &mut self.field;

        if false {
            return a;
        }

        return &self.field;
    }
}

// issue 124254
fn find_lowest_or_first_empty_pos(list: &mut [Option<u8>]) -> &mut Option<u8> {
    let mut low_pos_val: Option<(usize, u8)> = None;
    for (idx, i) in list.iter_mut().enumerate() {
        let Some(s) = i else {
            return i;
        };

        low_pos_val = match low_pos_val {
            Some((_oidx, oval)) if oval > *s => Some((idx, *s)),
            Some(old) => Some(old),
            None => Some((idx, *s)),
        };
    }
    let Some((lowest_idx, _)) = low_pos_val else {
        unreachable!("Can't have zero length list!");
    };
    &mut list[lowest_idx]
}

fn issue_124254() {
    let mut list = [Some(1), Some(2), None, Some(3)];
    let v = find_lowest_or_first_empty_pos(&mut list);
    assert!(v.is_none());
    assert_eq!(v as *mut _ as usize, list.as_ptr().wrapping_add(2) as usize);

    let mut list = [Some(1), Some(2), Some(3), Some(0)];
    let v = find_lowest_or_first_empty_pos(&mut list);
    assert_eq!(v, &mut Some(0));
    assert_eq!(v as *mut _ as usize, list.as_ptr().wrapping_add(3) as usize);

    println!("pass");
}

// issue 21906
struct A {
    a: i32,
}

impl A {
    fn one(&mut self) -> &i32 {
        self.a = 10;
        &self.a
    }
    fn two(&mut self) -> &i32 {
        loop {
            let k = self.one();
            if *k > 10i32 {
                return k;
            }
        }
    }
}

// issue 51545
fn borrow(o: &mut Option<i32>) -> Option<&mut i32> {
    match o.as_mut() {
        Some(i) => Some(i),
        None => o.as_mut(),
    }
}

// issue 58787

struct Node {
    rest: List,
}

struct List(Option<Box<Node>>);

fn issue_58787(arg: &mut List) {
    let mut list = arg;

    match list.0 {
        Some(ref mut d) => {
            if true {
                list = &mut d.rest;
            }
        }
        None => (),
    }

    match list.0 {
        Some(ref mut d) => {
            list = &mut d.rest;
        }
        None => (),
    }

    match list {
        List(Some(d)) => {
            if true {
                list = &mut d.rest;
            }
        }
        List(None) => (),
    }

    match list {
        List(Some(d)) => {
            list = &mut d.rest;
        }
        List(None) => (),
    }

    match &mut list.0 {
        Some(d) => {
            if true {
                list = &mut d.rest;
            }
        }
        None => (),
    }

    match &mut list.0 {
        Some(d) => {
            list = &mut d.rest;
        }
        None => (),
    }

    list.0 = None;
}

// issue 68934
enum Either<A,B> {
    Left(A),
    Right(B)
}

enum Tree<'a, A, B> {
    ALeaf(A),
    BLeaf(B),
    ABranch(&'a mut Tree<'a, A, B>, A),
    BBranch(&'a mut Tree<'a, A, B>, B)
}

impl<'a, A: PartialOrd, B> Tree<'a, A ,B> {
    fn deep_fetch(&mut self, value: Either<A, B>) -> Result<&mut Self, (&mut Self, Either<A,B>)> {
        match (self, value) {
            (Tree::ABranch(ref mut a, ref v), Either::Left(vv)) if v > &vv => {
                a.deep_fetch(Either::Left(vv))
            }

            (this, _v) => Err((this, _v))
        }
    }
}

fn main() {}
