// revisions: base extended

#![cfg_attr(extended, feature(generic_associated_types_extended))]
#![cfg_attr(extended, allow(incomplete_features))]

trait RefCont<'a, T> {
    fn t(&'a self) -> &'a T;
}

impl<'a, T> RefCont<'a, T> for &'a T {
    fn t(&'a self) -> &'a T {
        self
    }
}

impl<'a, T> RefCont<'a, T> for Box<T> {
    fn t(&'a self) -> &'a T {
        self.as_ref()
    }
}

trait MapLike<K, V> {
    type VRefCont<'a>: RefCont<'a, V> where Self: 'a;
    fn get<'a>(&'a self, key: &K) -> Option<Self::VRefCont<'a>>;
}

impl<K: Ord, V: 'static> MapLike<K, V> for std::collections::BTreeMap<K, V> {
    type VRefCont<'a> = &'a V where Self: 'a;
    fn get<'a>(&'a self, key: &K) -> Option<&'a V> {
        std::collections::BTreeMap::get(self, key)
    }
}

struct Source;

impl<K, V: Default> MapLike<K, V> for Source {
    type VRefCont<'a> = Box<V>;
    fn get<'a>(&self, _: &K) -> Option<Box<V>> {
        Some(Box::new(V::default()))
    }
}

fn main() {
    let m = Box::new(std::collections::BTreeMap::<u8, u8>::new())
    //[base]~^ ERROR the trait
    //[extended]~^^ type mismatch
        as Box<dyn MapLike<u8, u8, VRefCont = dyn RefCont<'_, u8>>>;
      //~^ ERROR missing generics for associated type
      //[base]~^^ ERROR the trait
}
