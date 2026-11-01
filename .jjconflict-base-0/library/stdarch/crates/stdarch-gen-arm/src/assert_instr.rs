use proc_macro2::TokenStream;
use quote::{ToTokens, TokenStreamExt, format_ident, quote};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, ser::SerializeSeq};
use std::fmt;

use crate::{
    context::{self, Context},
    typekinds::{BaseType, BaseTypeKind},
    wildstring::WildString,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InstructionAssertion {
    Basic(WildString),
    WithArgs(WildString, WildString),
}

impl InstructionAssertion {
    fn build(&mut self, ctx: &Context) -> context::Result {
        match self {
            InstructionAssertion::Basic(ws) => ws.build_acle(ctx.local),
            InstructionAssertion::WithArgs(ws, args_ws) => [ws, args_ws]
                .into_iter()
                .try_for_each(|ws| ws.build_acle(ctx.local)),
        }
    }
}

impl ToTokens for InstructionAssertion {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let instr = format_ident!(
            "{}",
            match self {
                Self::Basic(instr) => instr,
                Self::WithArgs(instr, _) => instr,
            }
            .to_string()
        );
        tokens.append_all(quote! { #instr });

        if let Self::WithArgs(_, args) = self {
            let ex: TokenStream = args
                .to_string()
                .parse()
                .expect("invalid instruction assertion arguments expression given");
            tokens.append_all(quote! {, #ex})
        }
    }
}

// Asserts that the given instruction is present for the intrinsic of the associated type bitsize.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(remote = "Self")]
pub struct InstructionAssertionMethodForBitsize {
    pub default: InstructionAssertion,
    pub byte: Option<InstructionAssertion>,
    pub halfword: Option<InstructionAssertion>,
    pub word: Option<InstructionAssertion>,
    pub doubleword: Option<InstructionAssertion>,
}

impl InstructionAssertionMethodForBitsize {
    fn build(&mut self, ctx: &Context) -> context::Result {
        if let Some(ref mut byte) = self.byte {
            byte.build(ctx)?
        }
        if let Some(ref mut halfword) = self.halfword {
            halfword.build(ctx)?
        }
        if let Some(ref mut word) = self.word {
            word.build(ctx)?
        }
        if let Some(ref mut doubleword) = self.doubleword {
            doubleword.build(ctx)?
        }
        self.default.build(ctx)
    }
}

impl Serialize for InstructionAssertionMethodForBitsize {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            InstructionAssertionMethodForBitsize {
                default: InstructionAssertion::Basic(instr),
                byte: None,
                halfword: None,
                word: None,
                doubleword: None,
            } => serializer.serialize_str(&instr.to_string()),
            InstructionAssertionMethodForBitsize {
                default: InstructionAssertion::WithArgs(instr, args),
                byte: None,
                halfword: None,
                word: None,
                doubleword: None,
            } => {
                let mut seq = serializer.serialize_seq(Some(2))?;
                seq.serialize_element(&instr.to_string())?;
                seq.serialize_element(&args.to_string())?;
                seq.end()
            }
            _ => InstructionAssertionMethodForBitsize::serialize(self, serializer),
        }
    }
}

impl<'de> Deserialize<'de> for InstructionAssertionMethodForBitsize {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct IAMVisitor;

        impl<'de> Visitor<'de> for IAMVisitor {
            type Value = InstructionAssertionMethodForBitsize;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("array, string or map")
            }

            fn visit_str<E>(self, value: &str) -> Result<InstructionAssertionMethodForBitsize, E>
            where
                E: de::Error,
            {
                Ok(InstructionAssertionMethodForBitsize {
                    default: InstructionAssertion::Basic(value.parse().map_err(E::custom)?),
                    byte: None,
                    halfword: None,
                    word: None,
                    doubleword: None,
                })
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                use serde::de::Error;
                let make_err =
                    || Error::custom("invalid number of arguments passed to assert_instruction");
                let instruction = seq.next_element()?.ok_or_else(make_err)?;
                let args = seq.next_element()?.ok_or_else(make_err)?;

                if let Some(true) = seq.size_hint().map(|len| len > 0) {
                    Err(make_err())
                } else {
                    Ok(InstructionAssertionMethodForBitsize {
                        default: InstructionAssertion::WithArgs(instruction, args),
                        byte: None,
                        halfword: None,
                        word: None,
                        doubleword: None,
                    })
                }
            }

            fn visit_map<M>(self, map: M) -> Result<InstructionAssertionMethodForBitsize, M::Error>
            where
                M: MapAccess<'de>,
            {
                InstructionAssertionMethodForBitsize::deserialize(
                    de::value::MapAccessDeserializer::new(map),
                )
            }
        }

        deserializer.deserialize_any(IAMVisitor)
    }
}

/// Asserts that the given instruction is present for the intrinsic of the associated type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(remote = "Self")]
pub struct InstructionAssertionMethod {
    /// Instruction for integer intrinsics
    pub default: InstructionAssertionMethodForBitsize,
    /// Instruction for floating-point intrinsics (optional)
    #[serde(default)]
    pub float: Option<InstructionAssertionMethodForBitsize>,
    /// Instruction for unsigned integer intrinsics (optional)
    #[serde(default)]
    pub unsigned: Option<InstructionAssertionMethodForBitsize>,
}

impl InstructionAssertionMethod {
    pub(crate) fn build(&mut self, ctx: &Context) -> context::Result {
        if let Some(ref mut float) = self.float {
            float.build(ctx)?
        }
        if let Some(ref mut unsigned) = self.unsigned {
            unsigned.build(ctx)?
        }
        self.default.build(ctx)
    }
}

impl Serialize for InstructionAssertionMethod {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            InstructionAssertionMethod {
                default:
                    InstructionAssertionMethodForBitsize {
                        default: InstructionAssertion::Basic(instr),
                        byte: None,
                        halfword: None,
                        word: None,
                        doubleword: None,
                    },
                float: None,
                unsigned: None,
            } => serializer.serialize_str(&instr.to_string()),
            InstructionAssertionMethod {
                default:
                    InstructionAssertionMethodForBitsize {
                        default: InstructionAssertion::WithArgs(instr, args),
                        byte: None,
                        halfword: None,
                        word: None,
                        doubleword: None,
                    },
                float: None,
                unsigned: None,
            } => {
                let mut seq = serializer.serialize_seq(Some(2))?;
                seq.serialize_element(&instr.to_string())?;
                seq.serialize_element(&args.to_string())?;
                seq.end()
            }
            _ => InstructionAssertionMethod::serialize(self, serializer),
        }
    }
}

impl<'de> Deserialize<'de> for InstructionAssertionMethod {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct IAMVisitor;

        impl<'de> Visitor<'de> for IAMVisitor {
            type Value = InstructionAssertionMethod;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("array, string or map")
            }

            fn visit_str<E>(self, value: &str) -> Result<InstructionAssertionMethod, E>
            where
                E: de::Error,
            {
                Ok(InstructionAssertionMethod {
                    default: InstructionAssertionMethodForBitsize {
                        default: InstructionAssertion::Basic(value.parse().map_err(E::custom)?),
                        byte: None,
                        halfword: None,
                        word: None,
                        doubleword: None,
                    },
                    float: None,
                    unsigned: None,
                })
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                use serde::de::Error;
                let make_err =
                    || Error::custom("invalid number of arguments passed to assert_instruction");
                let instruction = seq.next_element()?.ok_or_else(make_err)?;
                let args = seq.next_element()?.ok_or_else(make_err)?;

                if let Some(true) = seq.size_hint().map(|len| len > 0) {
                    Err(make_err())
                } else {
                    Ok(InstructionAssertionMethod {
                        default: InstructionAssertionMethodForBitsize {
                            default: InstructionAssertion::WithArgs(instruction, args),
                            byte: None,
                            halfword: None,
                            word: None,
                            doubleword: None,
                        },
                        float: None,
                        unsigned: None,
                    })
                }
            }

            fn visit_map<M>(self, map: M) -> Result<InstructionAssertionMethod, M::Error>
            where
                M: MapAccess<'de>,
            {
                InstructionAssertionMethod::deserialize(de::value::MapAccessDeserializer::new(map))
            }
        }

        deserializer.deserialize_any(IAMVisitor)
    }
}

#[derive(Debug)]
pub struct InstructionAssertionsForBaseType<'a>(
    pub &'a Vec<InstructionAssertionMethod>,
    pub &'a Option<&'a BaseType>,
);

impl<'a> ToTokens for InstructionAssertionsForBaseType<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.0.iter().for_each(
            |InstructionAssertionMethod {
                 default,
                 float,
                 unsigned,
             }| {
                let kind = self.1.map(|ty| ty.kind());
                let instruction = match (kind, float, unsigned) {
                    (None, float, unsigned) if float.is_some() || unsigned.is_some() => {
                        unreachable!(
                        "cannot determine the base type kind for instruction assertion: {self:#?}")
                    }
                    (Some(BaseTypeKind::Float), Some(float), _) => float,
                    (Some(BaseTypeKind::UInt), _, Some(unsigned)) => unsigned,
                    _ => default,
                };

                let bitsize = self.1.and_then(|ty| ty.get_size().ok());
                let instruction = match (bitsize, instruction) {
                    (
                        Some(8),
                        InstructionAssertionMethodForBitsize {
                            byte: Some(byte), ..
                        },
                    ) => byte,
                    (
                        Some(16),
                        InstructionAssertionMethodForBitsize {
                            halfword: Some(halfword),
                            ..
                        },
                    ) => halfword,
                    (
                        Some(32),
                        InstructionAssertionMethodForBitsize {
                            word: Some(word), ..
                        },
                    ) => word,
                    (
                        Some(64),
                        InstructionAssertionMethodForBitsize {
                            doubleword: Some(doubleword),
                            ..
                        },
                    ) => doubleword,
                    (_, InstructionAssertionMethodForBitsize { default, .. }) => default,
                };

                tokens.append_all(quote! { #[cfg_attr(test, assert_instr(#instruction))]})
            },
        );
    }
}
