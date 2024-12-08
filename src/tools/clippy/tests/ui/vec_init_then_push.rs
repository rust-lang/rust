#![allow(unused_variables)]
#![warn(clippy::vec_init_then_push)]
//@no-rustfix
fn main() {
    let mut def_err: Vec<u32> = Default::default();
    //~^ ERROR: calls to `push` immediately after creation
    //~| NOTE: `-D clippy::vec-init-then-push` implied by `-D warnings`
    def_err.push(0);

    let mut new_err = Vec::<u32>::new();
    //~^ ERROR: calls to `push` immediately after creation
    new_err.push(1);

    let mut cap_err = Vec::with_capacity(2);
    //~^ ERROR: calls to `push` immediately after creation
    cap_err.push(0);
    cap_err.push(1);
    cap_err.push(2);
    if true {
        // don't include this one
        cap_err.push(3);
    }

    let mut cap_ok = Vec::with_capacity(10);
    cap_ok.push(0);

    new_err = Vec::new();
    //~^ ERROR: calls to `push` immediately after creation
    new_err.push(0);

    let mut vec = Vec::new();
    // control flow at block final expression
    if true {
        // no lint
        vec.push(1);
    }

    let mut vec = Vec::with_capacity(5);
    vec.push(1);
    vec.push(2);
    vec.push(3);
    vec.push(4);
}

pub fn no_lint() -> Vec<i32> {
    let mut p = Some(1);
    let mut vec = Vec::new();
    loop {
        match p {
            None => return vec,
            Some(i) => {
                vec.push(i);
                p = None;
            },
        }
    }
}

fn _from_iter(items: impl Iterator<Item = u32>) -> Vec<u32> {
    let mut v = Vec::new();
    v.push(0);
    v.push(1);
    v.extend(items);
    v
}

fn _cond_push(x: bool) -> Vec<u32> {
    let mut v = Vec::new();
    v.push(0);
    if x {
        v.push(1);
    }
    v.push(2);
    v
}

fn _push_then_edit(x: u32) -> Vec<u32> {
    let mut v = Vec::new();
    //~^ ERROR: calls to `push` immediately after creation
    v.push(x);
    v.push(1);
    v[0] = v[1] + 5;
    v
}

fn _cond_push_with_large_start(x: bool) -> Vec<u32> {
    let mut v = Vec::new();
    //~^ ERROR: calls to `push` immediately after creation
    v.push(0);
    v.push(1);
    v.push(0);
    v.push(1);
    v.push(0);
    v.push(0);
    v.push(1);
    v.push(0);
    if x {
        v.push(1);
    }

    let mut v2 = Vec::new();
    //~^ ERROR: calls to `push` immediately after creation
    v2.push(0);
    v2.push(1);
    v2.push(0);
    v2.push(1);
    v2.push(0);
    v2.push(0);
    v2.push(1);
    v2.push(0);
    v2.extend(&v);

    v2
}

fn f() {
    let mut v = Vec::new();
    //~^ ERROR: calls to `push` immediately after creation
    v.push((0i32, 0i32));
    let y = v[0].0.abs();
}
