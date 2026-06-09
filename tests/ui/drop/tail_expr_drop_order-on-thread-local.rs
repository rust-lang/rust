//@ check-pass

#![feature(thread_local)]
#![deny(tail_expr_drop_order)]

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

pub struct Global;

#[thread_local]
static REENTRANCY_STATE: State<Global> = State { marker: PhantomData, controller: Global };

pub struct Token(PhantomData<*mut ()>);

pub fn with_mut<T>(f: impl FnOnce(&mut Token) -> T) -> T {
    f(&mut REENTRANCY_STATE.borrow_mut())
}

pub struct State<T: ?Sized = Global> {
    marker: PhantomData<*mut ()>,
    controller: T,
}

impl<T: ?Sized> State<T> {
    pub fn borrow_mut(&self) -> TokenMut<'_, T> {
        todo!()
    }
}

pub struct TokenMut<'a, T: ?Sized = Global> {
    state: &'a State<T>,
    token: Token,
}

impl<T> Deref for TokenMut<'_, T> {
    type Target = Token;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

impl<T> DerefMut for TokenMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        todo!()
    }
}

impl<T: ?Sized> Drop for TokenMut<'_, T> {
    fn drop(&mut self) {
        todo!()
    }
}

fn main() {}
