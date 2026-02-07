#![feature(arbitrary_self_types)]
#![deny(inherent_method_on_receiver)]
#![allow(dead_code)]

use std::ops::{Deref, Receiver};

// Base case to trigger the lint on `Deref`

pub struct Thing<T>(T);

impl<T> Deref for Thing<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> Thing<T> {
    pub fn method(&self) {}
    //~^ ERROR: inherent methods on types that implement `Deref` or `Receiver` shadow methods of their target [inherent_method_on_receiver]
}

// Base case to trigger the lint on `Receiver`

pub struct Thing2<T>(T);

impl<T> Receiver for Thing2<T> {
    type Target = T;
}

impl<T> Thing2<T> {
    pub fn method(&self) {}
    //~^ ERROR: inherent methods on types that implement `Deref` or `Receiver` shadow methods of their target [inherent_method_on_receiver]
}

// Also work with the `self: X` syntax

pub struct Thing3<T>(T);

impl<T> Receiver for Thing3<T> {
    type Target = T;
}

impl<T> Thing3<T> {
    pub fn method(self: Box<Self>) {}
    //~^ ERROR: inherent methods on types that implement `Deref` or `Receiver` shadow methods of their target [inherent_method_on_receiver]
}

// Not publicily accessible, no lint

pub struct Thing4<T>(T);

impl<T> Receiver for Thing4<T> {
    type Target = T;
}

impl<T> Thing4<T> {
    fn method(&self) {}
}

// Not publicily accessible, no lint 2

pub(crate) struct Thing5<T>(T);

impl<T> Receiver for Thing5<T> {
    type Target = T;
}

impl<T> Thing5<T> {
    fn method(&self) {}
}

// Only true generics are triggering the lint

pub(crate) struct Thing6<T>(T);

impl<T> Receiver for Thing6<T> {
    type Target = [T];
}

impl<T> Thing6<T> {
    pub fn method(&self) {}
}

// Receiving to something concrete, but that then derefs to a generic still triggers the lint

pub struct Thing7<T>(T);

impl<T> Receiver for Thing7<T> {
    type Target = Thing<T>;
}

impl<T> Thing7<T> {
    pub fn method(&self) {}
    //~^ ERROR: inherent methods on types that implement `Deref` or `Receiver` shadow methods of their target [inherent_method_on_receiver]
}

fn main() {}
