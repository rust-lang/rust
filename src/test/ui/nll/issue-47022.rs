// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![allow(warnings)]
#![feature(nll)]

struct LoadedObject {
    bodies: Vec<Body>,
    color: Color,
}

struct Body;

#[derive(Clone)]
struct Color;

struct Graphic {
    color: Color,
}

fn convert(objects: Vec<LoadedObject>) -> (Vec<Body>, Vec<Graphic>) {
    objects
        .into_iter()
        .flat_map(|LoadedObject { bodies, color, .. }| {
            bodies.into_iter().map(move |body| {
                (
                    body,
                    Graphic {
                        color: color.clone(),
                    },
                )
            })
        })
        .unzip()
}

fn main() {}

