// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::any::TypeId;
use std::rc::Rc;

type Fp<T> = Rc<T>;

struct Engine;

trait Component: 'static {}
impl Component for Engine {}

trait Env {
    fn get_component_type_id(&self, type_id: TypeId) -> Option<Fp<Component>>;
}

impl<'a> Env+'a {
    fn get_component<T: Component>(&self) -> Option<Fp<T>> {
        let x = self.get_component_type_id(TypeId::of::<T>());
        None
    }
}

trait Figment {
    fn init(&mut self, env: &Env);
}

struct MyFigment;

impl Figment for MyFigment {
    fn init(&mut self, env: &Env) {
        let engine = env.get_component::<Engine>();
    }
}

fn main() {}
