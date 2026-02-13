#![warn(clippy::manual_pop_if)]
#![allow(clippy::collapsible_if, clippy::redundant_closure)]

use std::collections::VecDeque;
use std::marker::PhantomData;

// FakeVec has the same methods as Vec but isn't actually a Vec
struct FakeVec<T>(PhantomData<T>);

impl<T> FakeVec<T> {
    fn last(&self) -> Option<&T> {
        None
    }

    fn pop(&mut self) -> Option<T> {
        None
    }
}

fn is_some_and_pattern_positive(mut vec: Vec<i32>, mut deque: VecDeque<i32>) {
    if vec.last().is_some_and(|x| *x > 2) {
        //~^ manual_pop_if
        vec.pop().unwrap();
    }

    if vec.last().is_some_and(|x| *x > 2) {
        //~^ manual_pop_if
        vec.pop().expect("element");
    }

    if deque.back().is_some_and(|x| *x > 2) {
        //~^ manual_pop_if
        deque.pop_back().unwrap();
    }

    if deque.front().is_some_and(|x| *x > 2) {
        //~^ manual_pop_if
        deque.pop_front().unwrap();
    }
}

fn is_some_and_pattern_negative(mut vec: Vec<i32>, mut deque: VecDeque<i32>) {
    // Do not lint, different vectors
    let mut vec2 = vec![0];
    if vec.last().is_some_and(|x| *x > 2) {
        vec2.pop().unwrap();
    }

    // Do not lint, non-Vec type
    let mut fake_vec: FakeVec<i32> = FakeVec(PhantomData);
    if fake_vec.last().is_some_and(|x| *x > 2) {
        fake_vec.pop().unwrap();
    }

    // Do not lint, else-if branch
    if false {
        // something
    } else if vec.last().is_some_and(|x| *x > 2) {
        vec.pop().unwrap();
    }

    // Do not lint, value used in let binding
    if vec.last().is_some_and(|x| *x > 2) {
        let _value = vec.pop().unwrap();
        println!("Popped: {}", _value);
    }

    // Do not lint, value used in expression
    if vec.last().is_some_and(|x| *x > 2) {
        println!("Popped: {}", vec.pop().unwrap());
    }

    // Do not lint, else block
    let _result = if vec.last().is_some_and(|x| *x > 2) {
        vec.pop().unwrap()
    } else {
        0
    };
}

fn if_let_pattern_positive(mut vec: Vec<i32>, mut deque: VecDeque<i32>) {
    if let Some(x) = vec.last() {
        //~^ manual_pop_if
        if *x > 2 {
            vec.pop().unwrap();
        }
    }

    if let Some(x) = vec.last() {
        //~^ manual_pop_if
        if *x > 2 {
            vec.pop().expect("element");
        }
    }

    if let Some(x) = deque.back() {
        //~^ manual_pop_if
        if *x > 2 {
            deque.pop_back().unwrap();
        }
    }

    if let Some(x) = deque.front() {
        //~^ manual_pop_if
        if *x > 2 {
            deque.pop_front().unwrap();
        }
    }
}

fn if_let_pattern_negative(mut vec: Vec<i32>) {
    // Do not lint, different vectors
    let mut vec2 = vec![0];
    if let Some(x) = vec.last() {
        if *x > 2 {
            vec2.pop().unwrap();
        }
    }

    // Do not lint, intervening statements
    if let Some(x) = vec.last() {
        println!("Checking {}", x);
        if *x > 2 {
            vec.pop().unwrap();
        }
    }

    // Do not lint, bound variable not used in condition
    if let Some(_x) = vec.last() {
        if vec.len() > 2 {
            vec.pop().unwrap();
        }
    }

    // Do not lint, value used in let binding
    if let Some(x) = vec.last() {
        if *x > 2 {
            let _val = vec.pop().unwrap();
        }
    }

    // Do not lint, else block
    let _result = if let Some(x) = vec.last() {
        if *x > 2 { vec.pop().unwrap() } else { 0 }
    } else {
        0
    };
}

fn let_chain_pattern_positive(mut vec: Vec<i32>, mut deque: VecDeque<i32>) {
    if let Some(x) = vec.last() //~ manual_pop_if
        && *x > 2
    {
        vec.pop().unwrap();
    }

    if let Some(x) = vec.last() //~ manual_pop_if
        && *x > 2
    {
        vec.pop().expect("element");
    }

    if let Some(x) = deque.back() //~ manual_pop_if
        && *x > 2
    {
        deque.pop_back().unwrap();
    }

    if let Some(x) = deque.front() //~ manual_pop_if
        && *x > 2
    {
        deque.pop_front().unwrap();
    }
}

fn let_chain_pattern_negative(mut vec: Vec<i32>) {
    // Do not lint, different vectors
    let mut vec2 = vec![0];
    if let Some(x) = vec.last()
        && *x > 2
    {
        vec2.pop().unwrap();
    }

    // Do not lint, bound variable not used in condition
    if let Some(_x) = vec.last()
        && vec.len() > 2
    {
        vec.pop().unwrap();
    }

    // Do not lint, value used in let binding
    if let Some(x) = vec.last()
        && *x > 2
    {
        let _val = vec.pop().unwrap();
    }

    // Do not lint, value used in expression
    if let Some(x) = vec.last()
        && *x > 2
    {
        println!("Popped: {}", vec.pop().unwrap());
    }

    // Do not lint, else block
    let _result = if let Some(x) = vec.last()
        && *x > 2
    {
        vec.pop().unwrap()
    } else {
        0
    };
}

fn map_unwrap_or_pattern_positive(mut vec: Vec<i32>, mut deque: VecDeque<i32>) {
    if vec.last().map(|x| *x > 2).unwrap_or(false) {
        //~^ manual_pop_if
        vec.pop().unwrap();
    }

    if vec.last().map(|x| *x > 2).unwrap_or(false) {
        //~^ manual_pop_if
        vec.pop().expect("element");
    }

    if deque.back().map(|x| *x > 2).unwrap_or(false) {
        //~^ manual_pop_if
        deque.pop_back().unwrap();
    }

    if deque.front().map(|x| *x > 2).unwrap_or(false) {
        //~^ manual_pop_if
        deque.pop_front().unwrap();
    }
}

fn map_unwrap_or_pattern_negative(mut vec: Vec<i32>) {
    // Do not lint, unwrap_or(true) instead of false
    if vec.last().map(|x| *x > 2).unwrap_or(true) {
        vec.pop().unwrap();
    }

    // Do not lint, different vectors
    let mut vec2 = vec![0];
    if vec.last().map(|x| *x > 2).unwrap_or(false) {
        vec2.pop().unwrap();
    }

    // Do not lint, non-Vec type
    let mut fake_vec: FakeVec<i32> = FakeVec(PhantomData);
    if fake_vec.last().map(|x| *x > 2).unwrap_or(false) {
        fake_vec.pop().unwrap();
    }

    // Do not lint, map returns non-boolean
    if vec.last().map(|x| x + 1).unwrap_or(0) > 2 {
        vec.pop().unwrap();
    }

    // Do not lint, value used in let binding
    if vec.last().map(|x| *x > 2).unwrap_or(false) {
        let _val = vec.pop().unwrap();
    }

    // Do not lint, else block
    let _result = if vec.last().map(|x| *x > 2).unwrap_or(false) {
        vec.pop().unwrap()
    } else {
        0
    };
}

// this makes sure we do not expand vec![] in the suggestion
fn handle_macro_in_closure(mut vec: Vec<Vec<i32>>) {
    if vec.last().is_some_and(|e| *e == vec![1]) {
        //~^ manual_pop_if
        vec.pop().unwrap();
    }
}

#[clippy::msrv = "1.85.0"]
fn msrv_too_low_vec(mut vec: Vec<i32>) {
    if vec.last().is_some_and(|x| *x > 2) {
        vec.pop().unwrap();
    }
}

#[clippy::msrv = "1.92.0"]
fn msrv_too_low_vecdeque(mut deque: VecDeque<i32>) {
    if deque.back().is_some_and(|x| *x > 2) {
        deque.pop_back().unwrap();
    }

    if deque.front().is_some_and(|x| *x > 2) {
        deque.pop_front().unwrap();
    }
}

#[clippy::msrv = "1.86.0"]
fn msrv_high_enough_vec(mut vec: Vec<i32>) {
    if vec.last().is_some_and(|x| *x > 2) {
        //~^ manual_pop_if
        vec.pop().unwrap();
    }
}

#[clippy::msrv = "1.93.0"]
fn msrv_high_enough_vecdeque(mut deque: VecDeque<i32>) {
    if deque.back().is_some_and(|x| *x > 2) {
        //~^ manual_pop_if
        deque.pop_back().unwrap();
    }

    if deque.front().is_some_and(|x| *x > 2) {
        //~^ manual_pop_if
        deque.pop_front().unwrap();
    }
}
