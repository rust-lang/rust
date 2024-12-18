use proc_macro2::TokenStream;
use quote::ToTokens;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::context::{self, LocalContext};
use crate::typekinds::{BaseType, BaseTypeKind, TypeKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchSizeValues<T> {
    pub default: T,
    pub byte: Option<T>,
    pub halfword: Option<T>,
    pub doubleword: Option<T>,
}

impl<T> MatchSizeValues<T> {
    pub fn get(&mut self, ty: &TypeKind, ctx: &LocalContext) -> context::Result<&T> {
        let base_ty = if let Some(w) = ty.wildcard() {
            ctx.provide_type_wildcard(w)?
        } else {
            ty.clone()
        };

        if let BaseType::Sized(_, bitsize) = base_ty.base_type().unwrap() {
            match (bitsize, &self.byte, &self.halfword, &self.doubleword) {
                (64, _, _, Some(v)) | (16, _, Some(v), _) | (8, Some(v), _, _) => Ok(v),
                _ => Ok(&self.default),
            }
        } else {
            Err(format!("cannot match bitsize to unsized type {ty:?}!"))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchKindValues<T> {
    pub default: T,
    pub float: Option<T>,
    pub unsigned: Option<T>,
}

impl<T> MatchKindValues<T> {
    pub fn get(&mut self, ty: &TypeKind, ctx: &LocalContext) -> context::Result<&T> {
        let base_ty = if let Some(w) = ty.wildcard() {
            ctx.provide_type_wildcard(w)?
        } else {
            ty.clone()
        };

        match (
            base_ty.base_type().unwrap().kind(),
            &self.float,
            &self.unsigned,
        ) {
            (BaseTypeKind::Float, Some(v), _) | (BaseTypeKind::UInt, _, Some(v)) => Ok(v),
            _ => Ok(&self.default),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum SizeMatchable<T> {
    Matched(T),
    Unmatched {
        match_size: Option<TypeKind>,
        #[serde(flatten)]
        values: MatchSizeValues<Box<T>>,
    },
}

impl<T: Clone> SizeMatchable<T> {
    pub fn perform_match(&mut self, ctx: &LocalContext) -> context::Result {
        match self {
            Self::Unmatched {
                match_size: None,
                values: MatchSizeValues { default, .. },
            } => *self = Self::Matched(*default.to_owned()),
            Self::Unmatched {
                match_size: Some(ty),
                values,
            } => *self = Self::Matched(*values.get(ty, ctx)?.to_owned()),
            _ => {}
        }
        Ok(())
    }
}

impl<T: fmt::Debug> AsRef<T> for SizeMatchable<T> {
    fn as_ref(&self) -> &T {
        if let SizeMatchable::Matched(v) = self {
            v
        } else {
            panic!("no match for {self:?} was performed");
        }
    }
}

impl<T: fmt::Debug> AsMut<T> for SizeMatchable<T> {
    fn as_mut(&mut self) -> &mut T {
        if let SizeMatchable::Matched(v) = self {
            v
        } else {
            panic!("no match for {self:?} was performed");
        }
    }
}

impl<T: fmt::Debug + ToTokens> ToTokens for SizeMatchable<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.as_ref().to_tokens(tokens)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum KindMatchable<T> {
    Matched(T),
    Unmatched {
        match_kind: Option<TypeKind>,
        #[serde(flatten)]
        values: MatchKindValues<Box<T>>,
    },
}

impl<T: Clone> KindMatchable<T> {
    pub fn perform_match(&mut self, ctx: &LocalContext) -> context::Result {
        match self {
            Self::Unmatched {
                match_kind: None,
                values: MatchKindValues { default, .. },
            } => *self = Self::Matched(*default.to_owned()),
            Self::Unmatched {
                match_kind: Some(ty),
                values,
            } => *self = Self::Matched(*values.get(ty, ctx)?.to_owned()),
            _ => {}
        }
        Ok(())
    }
}

impl<T: fmt::Debug> AsRef<T> for KindMatchable<T> {
    fn as_ref(&self) -> &T {
        if let KindMatchable::Matched(v) = self {
            v
        } else {
            panic!("no match for {self:?} was performed");
        }
    }
}

impl<T: fmt::Debug> AsMut<T> for KindMatchable<T> {
    fn as_mut(&mut self) -> &mut T {
        if let KindMatchable::Matched(v) = self {
            v
        } else {
            panic!("no match for {self:?} was performed");
        }
    }
}

impl<T: fmt::Debug + ToTokens> ToTokens for KindMatchable<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.as_ref().to_tokens(tokens)
    }
}
