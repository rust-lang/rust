use std::borrow::Cow;
use std::collections::BTreeMap;

pub use serde_json::Value as Json;
use serde_json::{Map, Number, json};

use crate::spec::TargetMetadata;

pub trait ToJson {
    fn to_json(&self) -> Json;
}

impl ToJson for Json {
    fn to_json(&self) -> Json {
        self.clone()
    }
}

macro_rules! to_json_impl_num {
    ($($t:ty), +) => (
        $(impl ToJson for $t {
            fn to_json(&self) -> Json {
                Json::Number(Number::from(*self))
            }
        })+
    )
}

to_json_impl_num! { isize, i8, i16, i32, i64, usize, u8, u16, u32, u64 }

impl ToJson for bool {
    fn to_json(&self) -> Json {
        Json::Bool(*self)
    }
}

impl ToJson for str {
    fn to_json(&self) -> Json {
        Json::String(self.to_owned())
    }
}

impl ToJson for String {
    fn to_json(&self) -> Json {
        Json::String(self.to_owned())
    }
}

impl<'a> ToJson for Cow<'a, str> {
    fn to_json(&self) -> Json {
        Json::String(self.to_string())
    }
}

impl<A: ToJson> ToJson for [A] {
    fn to_json(&self) -> Json {
        Json::Array(self.iter().map(|elt| elt.to_json()).collect())
    }
}

impl<A: ToJson> ToJson for Vec<A> {
    fn to_json(&self) -> Json {
        Json::Array(self.iter().map(|elt| elt.to_json()).collect())
    }
}

impl<'a, A: ToJson> ToJson for Cow<'a, [A]>
where
    [A]: ToOwned,
{
    fn to_json(&self) -> Json {
        Json::Array(self.iter().map(|elt| elt.to_json()).collect())
    }
}

impl<T: ToString, A: ToJson> ToJson for BTreeMap<T, A> {
    fn to_json(&self) -> Json {
        let mut d = Map::new();
        for (key, value) in self {
            d.insert(key.to_string(), value.to_json());
        }
        Json::Object(d)
    }
}

impl<A: ToJson> ToJson for Option<A> {
    fn to_json(&self) -> Json {
        match *self {
            None => Json::Null,
            Some(ref value) => value.to_json(),
        }
    }
}

impl ToJson for TargetMetadata {
    fn to_json(&self) -> Json {
        json!({
            "description": self.description,
            "tier": self.tier,
            "host_tools": self.host_tools,
            "std": self.std,
        })
    }
}

impl ToJson for rustc_abi::Endian {
    fn to_json(&self) -> Json {
        self.as_str().to_json()
    }
}

impl ToJson for rustc_abi::CanonAbi {
    fn to_json(&self) -> Json {
        self.to_string().to_json()
    }
}
