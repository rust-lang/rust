// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(unused_variables)]
use std::any::TypeId;
use std::rc::Rc;

type Fp<T> = Rc<T>;

struct Engine;

trait Component: 'static {}
impl Component for Engine {}

trait Env {
    fn get_component_type_id(&self, type_id: TypeId) -> Option<Fp<dyn Component>>;
}

impl<'a> dyn Env + 'a {
    fn get_component<T: Component>(&self) -> Option<Fp<T>> {
        let x = self.get_component_type_id(TypeId::of::<T>());
        None
    }
}

trait Figment {
    fn init(&mut self, env: &dyn Env);
}

struct MyFigment;

impl Figment for MyFigment {
    fn init(&mut self, env: &dyn Env) {
        let engine = env.get_component::<Engine>();
    }
}

fn main() {}
