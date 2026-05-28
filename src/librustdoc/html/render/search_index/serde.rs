use std::fmt::{self, Formatter};

use rustc_span::Symbol;
use serde::de::{self, SeqAccess};
use serde::ser::SerializeSeq as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::html::render::IndexItemFunctionType;

impl Serialize for IndexItemFunctionType {
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
                    self.0
                        .iter()
                        .map(|symbol| symbol.as_ref().map(Symbol::as_str).unwrap_or_default()),
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

impl<'de> Deserialize<'de> for IndexItemFunctionType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Deserialized {
            #[serde(deserialize_with = "function_signature")]
            function_signature: IndexItemFunctionType,
            #[serde(deserialize_with = "param_names")]
            param_names: Vec<Option<Symbol>>,
        }

        fn function_signature<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<IndexItemFunctionType, D::Error> {
            String::deserialize(deserializer).map(|sig| {
                IndexItemFunctionType::read_from_string_without_param_names(sig.as_bytes()).0
            })
        }

        fn param_names<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<Vec<Option<Symbol>>, D::Error> {
            struct Visitor;

            impl<'de> de::Visitor<'de> for Visitor {
                type Value = Vec<Option<Symbol>>;

                fn expecting(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    f.write_str("seq of param names")
                }

                fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                where
                    A: SeqAccess<'de>,
                {
                    let mut param_names = Vec::with_capacity(seq.size_hint().unwrap_or_default());

                    while let Some(symbol) = seq.next_element::<String>()? {
                        param_names.push(if symbol.is_empty() {
                            None
                        } else {
                            Some(Symbol::intern(&symbol))
                        });
                    }

                    Ok(param_names)
                }
            }

            deserializer.deserialize_seq(Visitor)
        }

        let Deserialized { mut function_signature, param_names } =
            Deserialized::deserialize(deserializer)?;
        function_signature.param_names = param_names;

        Ok(function_signature)
    }
}
