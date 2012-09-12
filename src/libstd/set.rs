import dvec::dvec;
import map::{hashfn, eqfn, hashmap};

struct set<K: copy> {
    mut implementation: option<set_implementation<K>>
}

struct list_set<K> {
    hasher: hashfn<K>;
    eqer: eqfn<K>;
    elements: ~[K];
}

enum set_implementation<K: copy> {
    impl_with_list(list_set<K>),
    impl_with_map(hashmap<K, ()>)
}

const threshold: uint = 25; // completely arbitrary.

impl<K> &list_set {
    pure fn contains(element: &K) {
        for self.elements.each |existing_element| {
            if self.eqer(element, existing_element) {
                return true;
            }
        }
        return false;
    }

    pure fn convert_to_map() -> hashmap<K, ()> {
        ...
    }
}

impl<K: copy> set<K> {
    fn add(+element: K) -> bool {
        let mut set_impl = option::swap_unwrap(&mut self.implementation);
        let contained_before = match set_impl {
          impl_with_list(ref mut list_set) => {
            if list_set.elements.len() >= threshold {
                // convert to a map
                self.implementation = some(list_set.convert_to_map());
                return self.add(move element);
            }

            if list_set.contains(&element) {
                false
            } else {
                vec::push(list_set.elements, element);
                true
            }
          }

          impl_with_map(ref map) => {
            let contained_before = map.insert(element, ());
          }
        }
        self.implementation = some(move set_impl);
        return true;
    }
}