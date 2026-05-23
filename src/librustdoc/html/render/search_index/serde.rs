use std::alloc::Allocator;
use std::fmt;

use rustc_span::Symbol;
use serde::de::{self, Error, MapAccess};
use serde::ser::SerializeSeq as _;
use serde::{Deserializer, Serialize, Serializer};
use serde_alloc::{DeserializeWithAlloc, WithAllocSeed};

use crate::html::render::search_index::SerializedSearchIndex;
use crate::html::render::{CrateInfo, IndexItemFunctionType};

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

// struct Deserialized<A: Allocator + Copy> {
//     function_signature: IndexItemFunctionType<A>,
//     param_names: Vec<Option<rustc_span::Symbol>>,
// }

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

                        // FIXME: should be able to work on &str
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

impl<'de, A: Allocator + Copy> DeserializeWithAlloc<'de, A> for SerializedSearchIndex<A> {
    fn deserialize_with_alloc<D>(deserializer: D, alloc: A) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Visitor<A: Allocator + Copy> {
            alloc: A,
        }

        impl<'de, A: Allocator + Copy> de::Visitor<'de> for Visitor<A> {
            type Value = SerializedSearchIndex<A>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("SerializedSearchIndex")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut names = None;
                let mut path_data = None;
                let mut entry_data = None;
                let mut descs = None;
                let mut function_data = None;
                let mut alias_pointers = None;
                let mut type_data = None;
                let mut generic_inverted_index = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "names" => {
                            if names.is_some() {
                                return Err(de::Error::duplicate_field("names"));
                            }
                            names = Some(map.next_value()?);
                        }
                        "path_data" => {
                            if path_data.is_some() {
                                return Err(de::Error::duplicate_field("path_data"));
                            }
                            path_data = Some(map.next_value()?);
                        }
                        "entry_data" => {
                            if entry_data.is_some() {
                                return Err(de::Error::duplicate_field("entry_data"));
                            }
                            entry_data = Some(map.next_value()?);
                        }
                        "descs" => {
                            if descs.is_some() {
                                return Err(de::Error::duplicate_field("descs"));
                            }
                            descs = Some(map.next_value()?);
                        }
                        "function_data" => {
                            if function_data.is_some() {
                                return Err(de::Error::duplicate_field("function_data"));
                            }
                            function_data = Some(map.next_value_seed::<WithAllocSeed<
                                Vec<Option<IndexItemFunctionType<A>>, A>,
                                A,
                            >>(
                                WithAllocSeed::new(self.alloc)
                            )?);
                        }
                        "alias_pointers" => {
                            if alias_pointers.is_some() {
                                return Err(de::Error::duplicate_field("alias_pointers"));
                            }
                            alias_pointers = Some(map.next_value()?);
                        }
                        "type_data" => {
                            if type_data.is_some() {
                                return Err(de::Error::duplicate_field("type_data"));
                            }
                            type_data = Some(map.next_value()?);
                        }
                        "generic_inverted_index" => {
                            if generic_inverted_index.is_some() {
                                return Err(de::Error::duplicate_field("generic_inverted_index"));
                            }
                            generic_inverted_index = Some(map.next_value()?);
                        }
                        _ => {
                            return Err(V::Error::unknown_field(
                                key,
                                &[
                                    "names",
                                    "path_data",
                                    "entry_data",
                                    "descs",
                                    "function_data",
                                    "alias_pointers",
                                    "type_data",
                                    "generic_inverted_index",
                                ],
                            ));
                        }
                    }
                }

                Ok(Self::Value {
                    names: names.unwrap(),
                    path_data: path_data.unwrap(),
                    entry_data: entry_data.unwrap(),
                    descs: descs.unwrap(),
                    function_data: function_data.unwrap(),
                    alias_pointers: alias_pointers.unwrap(),
                    type_data: type_data.unwrap(),
                    generic_inverted_index: generic_inverted_index.unwrap(),
                    crate_paths_index: Default::default(),
                })
            }
        }

        deserializer.deserialize_any(Visitor { alloc })
    }
}

impl<'de, A: Allocator + Copy> DeserializeWithAlloc<'de, A> for CrateInfo<A> {
    fn deserialize_with_alloc<D>(deserializer: D, alloc: A) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Visitor<A: Allocator + Copy> {
            alloc: A,
        }

        impl<'de, A: Allocator + Copy> de::Visitor<'de> for Visitor<A> {
            type Value = CrateInfo<A>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("CrateInfo struct")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut version = None;
                let mut src_files_js = None;
                let mut search_index = None;
                let mut all_crates = None;
                let mut crates_index = None;
                let mut trait_impl = None;
                let mut type_impl = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "version" => {
                            if version.is_some() {
                                return Err(de::Error::duplicate_field("version"));
                            }
                            version = Some(map.next_value()?);
                        }
                        "src_files_js" => {
                            if src_files_js.is_some() {
                                return Err(de::Error::duplicate_field("src_files_js"));
                            }
                            src_files_js = Some(map.next_value()?);
                        }
                        "search_index" => {
                            if search_index.is_some() {
                                return Err(de::Error::duplicate_field("search_index"));
                            }
                            search_index =
                                Some(map.next_value_seed(WithAllocSeed::new(self.alloc))?);
                        }
                        "all_crates" => {
                            if all_crates.is_some() {
                                return Err(de::Error::duplicate_field("all_crates"));
                            }
                            all_crates = Some(map.next_value()?);
                        }
                        "crates_index" => {
                            if crates_index.is_some() {
                                return Err(de::Error::duplicate_field("crates_index"));
                            }
                            crates_index = Some(map.next_value()?);
                        }
                        "trait_impl" => {
                            if trait_impl.is_some() {
                                return Err(de::Error::duplicate_field("trait_impl"));
                            }
                            trait_impl = Some(map.next_value()?);
                        }
                        "type_impl" => {
                            if type_impl.is_some() {
                                return Err(de::Error::duplicate_field("type_impl"));
                            }
                            type_impl = Some(map.next_value()?);
                        }
                        _ => {
                            return Err(M::Error::unknown_field(
                                key,
                                &[
                                    "version",
                                    "src_files_js",
                                    "search_index",
                                    "all_crates",
                                    "crates_index",
                                    "trait_impl",
                                    "type_impl",
                                ],
                            ));
                        }
                    }
                }

                Ok(Self::Value {
                    version: version.unwrap(),
                    src_files_js: src_files_js.unwrap(),
                    search_index: search_index.unwrap(),
                    all_crates: all_crates.unwrap(),
                    crates_index: crates_index.unwrap(),
                    trait_impl: trait_impl.unwrap(),
                    type_impl: type_impl.unwrap(),
                })
            }
        }

        deserializer.deserialize_map(Visitor { alloc })
    }
}
