use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;

use crate::core::builder::Step;

pub type Interned<T> = internment::Intern<T>;

/// This is essentially a `HashMap` which allows storing any type in its input and
/// any type in its output. It is a write-once cache; values are never evicted,
/// which means that references to the value can safely be returned from the
/// `get()` method.
#[derive(Debug)]
pub struct Cache(
    RefCell<
        HashMap<
            TypeId,
            Box<dyn Any>, // actually a HashMap<Step, Interned<Step::Output>>
        >,
    >,
);

impl Cache {
    pub fn new() -> Cache {
        Cache(RefCell::new(HashMap::new()))
    }

    pub fn put<S: Step>(&self, step: S, value: S::Output) {
        let mut cache = self.0.borrow_mut();
        let type_id = TypeId::of::<S>();
        let stepcache = cache
            .entry(type_id)
            .or_insert_with(|| Box::<HashMap<S, S::Output>>::default())
            .downcast_mut::<HashMap<S, S::Output>>()
            .expect("invalid type mapped");
        assert!(!stepcache.contains_key(&step), "processing {step:?} a second time");
        stepcache.insert(step, value);
    }

    pub fn get<S: Step>(&self, step: &S) -> Option<S::Output> {
        let mut cache = self.0.borrow_mut();
        let type_id = TypeId::of::<S>();
        let stepcache = cache
            .entry(type_id)
            .or_insert_with(|| Box::<HashMap<S, S::Output>>::default())
            .downcast_mut::<HashMap<S, S::Output>>()
            .expect("invalid type mapped");
        stepcache.get(step).cloned()
    }
}

#[cfg(test)]
impl Cache {
    pub fn all<S: Ord + Clone + Step>(&mut self) -> Vec<(S, S::Output)> {
        let cache = self.0.get_mut();
        let type_id = TypeId::of::<S>();
        let mut v = cache
            .remove(&type_id)
            .map(|b| b.downcast::<HashMap<S, S::Output>>().expect("correct type"))
            .map(|m| m.into_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        v.sort_by_key(|(s, _)| s.clone());
        v
    }

    pub fn contains<S: Step>(&self) -> bool {
        self.0.borrow().contains_key(&TypeId::of::<S>())
    }
}
