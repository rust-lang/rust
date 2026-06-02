use std::alloc::Allocator;
use std::fmt;

use rustc_span::Symbol;
use serde::de::{self, Error};
use serde::ser::SerializeSeq as _;
use serde::{Deserializer, Serialize, Serializer};
use serde_alloc::{DeserializeWithAlloc, WithAllocSeed};

use crate::html::render::IndexItemFunctionType;

impl<A: Allocator + Copy> Serialize for IndexItemFunctionType<A> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        struct ParamNames<'a>(&'a [Option<Symbol>]);

        impl<'a> Serialize for ParamNames<'a> {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.collect_seq(
                    self.0.iter().map(|sym| sym.as_ref().map(Symbol::as_str).unwrap_or_default()),
                )
            }
        }

        let mut seq = serializer.serialize_seq(Some(2))?;

        let mut fn_type = String::new();
        self.write_to_string_without_param_names(&mut fn_type);
        seq.serialize_element(&fn_type)?;

        seq.serialize_element(&ParamNames(&self.param_names))?;

        seq.end()
    }
}

impl<'de, A: Allocator + Copy> DeserializeWithAlloc<'de, A> for IndexItemFunctionType<A> {
    fn deserialize_with_alloc<D>(deserializer: D, alloc: A) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Blah<A: Allocator + Copy>(IndexItemFunctionType<A>);

        impl<'de, A: Allocator + Copy> DeserializeWithAlloc<'de, A> for Blah<A> {
            fn deserialize_with_alloc<D>(deserializer: D, alloc: A) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct Visitor<A: Allocator + Copy> {
                    alloc: A,
                }

                impl<'de, A: Allocator + Copy> de::Visitor<'de> for Visitor<A> {
                    type Value = IndexItemFunctionType<A>;

                    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                        formatter.write_str("IndexItemFunctionType")
                    }

                    fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
                    where
                        E: de::Error,
                    {
                        let (ty, _) = IndexItemFunctionType::read_from_string_without_param_names(
                            v, self.alloc,
                        );

                        Ok(ty)
                    }

                    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                    where
                        E: de::Error,
                    {
                        self.visit_bytes(v.as_bytes())
                    }
                }

                deserializer.deserialize_any(Visitor { alloc }).map(Self)
            }
        }

        struct ParamNames<A: Allocator + Copy>(Vec<Option<Symbol>, A>);

        impl<'de, A: Allocator + Copy> DeserializeWithAlloc<'de, A> for ParamNames<A> {
            fn deserialize_with_alloc<D>(deserializer: D, alloc: A) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct Visitor<A: Allocator + Copy> {
                    alloc: A,
                }

                impl<'de, A: Allocator + Copy> de::Visitor<'de> for Visitor<A> {
                    type Value = Vec<Option<Symbol>, A>;

                    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                        formatter.write_str("sequence of symbols")
                    }

                    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
                    where
                        S: de::SeqAccess<'de>,
                    {
                        let mut vec =
                            Vec::with_capacity_in(seq.size_hint().unwrap_or_default(), self.alloc);

                        // FIXME(yotamofek): should be able to work on &str
                        while let Some(sym) = seq.next_element::<Option<String>>()? {
                            vec.push(sym.map(|sym| Symbol::intern(&sym)));
                        }

                        Ok(vec)
                    }
                }

                deserializer.deserialize_seq(Visitor { alloc }).map(Self)
            }
        }

        struct Visitor<A: Allocator + Copy> {
            alloc: A,
        }

        impl<'de, A: Allocator + Copy> de::Visitor<'de> for Visitor<A> {
            type Value = IndexItemFunctionType<A>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("sequence of index item function type and param names")
            }

            fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
            where
                S: de::SeqAccess<'de>,
            {
                let Blah(mut ty) = seq
                    .next_element_seed(WithAllocSeed::new(self.alloc))?
                    .ok_or_else(|| S::Error::missing_field("index item function type"))?;

                ParamNames(ty.param_names) = seq
                    .next_element_seed(WithAllocSeed::new(self.alloc))?
                    .ok_or_else(|| S::Error::missing_field("param names"))?;

                Ok(ty)
            }
        }

        deserializer.deserialize_any(Visitor { alloc })
    }
}
