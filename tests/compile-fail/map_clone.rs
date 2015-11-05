#![feature(plugin)]

#![plugin(clippy)]
#![deny(map_clone)]

#![allow(unused)]

use std::ops::Deref;

fn map_clone_iter() {
    let x = [1,2,3];
    x.iter().map(|y| y.clone()); //~ ERROR you seem to be using .map()
                                 //~^ HELP try
    x.iter().map(|&y| y); //~ ERROR you seem to be using .map()
                          //~^ HELP try
    x.iter().map(|y| *y); //~ ERROR you seem to be using .map()
                          //~^ HELP try
    x.iter().map(Clone::clone); //~ ERROR you seem to be using .map()
                                //~^ HELP try
}

fn map_clone_option() {
    let x = Some(4);
    x.as_ref().map(|y| y.clone()); //~ ERROR you seem to be using .map()
                                   //~^ HELP try
    x.as_ref().map(|&y| y); //~ ERROR you seem to be using .map()
                            //~^ HELP try
    x.as_ref().map(|y| *y); //~ ERROR you seem to be using .map()
                            //~^ HELP try
}

fn not_linted_option() {
    let x = Some(5);

    // Not linted: other statements
    x.as_ref().map(|y| {
        println!("y: {}", y);
        y.clone()
    });

    // Not linted: argument bindings
    let x = Some((6, 7));
    x.map(|(y, _)| y.clone());

    // Not linted: cloning something else
    x.map(|y| y.0.clone());

    // Not linted: no dereferences
    x.map(|y| y);

    // Not linted: multiple dereferences
    let _: Option<(i32, i32)> = x.as_ref().as_ref().map(|&&x| x);
}

#[derive(Copy, Clone)]
struct Wrapper<T>(T);
impl<T> Wrapper<T> {
    fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Wrapper<U> {
        Wrapper(f(self.0))
    }
}

fn map_clone_other() {
    let eight = 8;
    let x = Wrapper(&eight);

    // Not linted: not a linted type
    x.map(|y| y.clone());
    x.map(|&y| y);
    x.map(|y| *y);
}

#[derive(Copy, Clone)]
struct UnusualDeref;
static NINE: i32 = 9;

impl Deref for UnusualDeref {
    type Target = i32;
    fn deref(&self) -> &i32 { &NINE }
}

fn map_clone_deref() {
    let x = Some(UnusualDeref);
    let _: Option<UnusualDeref> = x.as_ref().map(|y| *y); //~ ERROR you seem to be using .map()
                                                          //~^ HELP try

    // Not linted: using deref conversion
    let _: Option<i32> = x.map(|y| *y);

    // Not linted: using regular deref but also deref conversion
    let _: Option<i32> = x.as_ref().map(|y| **y);
}

fn main() { }
