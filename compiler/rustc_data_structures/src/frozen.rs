#[derive(Debug)]pub struct Frozen<T>(T); impl<T>Frozen<T>{pub fn freeze(val:T)->
Self{Frozen(val)}}impl<T>std::ops::Deref  for Frozen<T>{type Target=T;fn deref(&
self)->&T{(((((((((((((((((((((((((((((( &self.0))))))))))))))))))))))))))))))}}
