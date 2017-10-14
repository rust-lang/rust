
extern crate serde;

use self::serde::ser::{Serialize, Serializer, SerializeMap};
use self::serde::de::{Deserialize, Deserializer, MapAccess, Visitor};

use std::fmt::{self, Formatter};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use OrderMap;

/// Requires crate feature `"serde-1"`
impl<K, V, S> Serialize for OrderMap<K, V, S>
    where K: Serialize + Hash + Eq,
          V: Serialize,
          S: BuildHasher
{
    fn serialize<T>(&self, serializer: T) -> Result<T::Ok, T::Error>
        where T: Serializer
    {
        let mut map_serializer = try!(serializer.serialize_map(Some(self.len())));
        for (key, value) in self {
            try!(map_serializer.serialize_entry(key, value));
        }
        map_serializer.end()
    }
}

struct OrderMapVisitor<K, V, S>(PhantomData<(K, V, S)>);

impl<'de, K, V, S> Visitor<'de> for OrderMapVisitor<K, V, S>
    where K: Deserialize<'de> + Eq + Hash,
          V: Deserialize<'de>,
          S: Default + BuildHasher
{
    type Value = OrderMap<K, V, S>;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        write!(formatter, "a map")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where A: MapAccess<'de>
    {
        let mut values = OrderMap::with_capacity_and_hasher(map.size_hint().unwrap_or(0), Default::default());

        while let Some((key, value)) = try!(map.next_entry()) {
            values.insert(key, value);
        }

        Ok(values)
    }
}

/// Requires crate feature `"serde-1"`
impl<'de, K, V, S> Deserialize<'de> for OrderMap<K, V, S>
    where K: Deserialize<'de> + Eq + Hash,
          V: Deserialize<'de>,
          S: Default + BuildHasher
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        deserializer.deserialize_map(OrderMapVisitor(PhantomData))
    }
}
