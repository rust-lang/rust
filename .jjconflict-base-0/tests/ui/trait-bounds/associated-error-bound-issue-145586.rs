//@ run-rustfix

#![allow(dead_code)]

use std::marker::PhantomData;

trait Visitor<'de> {
    type Value;
}

trait Deserializer<'de> {
    type Error;

    fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>;
}

struct Wrapper<'de, T, E>(Result<T, E>, PhantomData<&'de ()>);

impl<'de, T, E> Deserializer<'de> for Wrapper<'de, T, E>
where
    T: Deserializer<'de>,
{
    type Error = E;

    fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        match self.0 {
            Ok(deserializer) => deserializer.deserialize_ignored_any(visitor), //~ ERROR mismatched types
            Err(error) => Err(error),
        }
    }
}

fn main() {}
