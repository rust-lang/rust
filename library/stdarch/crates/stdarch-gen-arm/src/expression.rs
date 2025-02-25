use itertools::Itertools;
use lazy_static::lazy_static;
use proc_macro2::{Literal, Punct, Spacing, TokenStream};
use quote::{ToTokens, TokenStreamExt, format_ident, quote};
use regex::Regex;
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::intrinsic::Intrinsic;
use crate::wildstring::WildStringPart;
use crate::{
    context::{self, Context, VariableType},
    intrinsic::{Argument, LLVMLink, StaticDefinition},
    matching::{MatchKindValues, MatchSizeValues},
    typekinds::{BaseType, BaseTypeKind, TypeKind},
    wildcards::Wildcard,
    wildstring::WildString,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IdentifierType {
    Variable,
    Symbol,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LetVariant {
    Basic(WildString, Box<Expression>),
    WithType(WildString, TypeKind, Box<Expression>),
    MutWithType(WildString, TypeKind, Box<Expression>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnCall(
    /// Function pointer
    pub Box<Expression>,
    /// Function arguments
    pub Vec<Expression>,
    /// Function turbofish arguments
    #[serde(default)]
    pub Vec<Expression>,
    /// Function requires unsafe wrapper
    #[serde(default)]
    pub bool,
);

impl FnCall {
    pub fn new_expression(fn_ptr: Expression, arguments: Vec<Expression>) -> Expression {
        FnCall(Box::new(fn_ptr), arguments, Vec::new(), false).into()
    }

    pub fn new_unsafe_expression(fn_ptr: Expression, arguments: Vec<Expression>) -> Expression {
        FnCall(Box::new(fn_ptr), arguments, Vec::new(), true).into()
    }

    pub fn is_llvm_link_call(&self, llvm_link_name: &String) -> bool {
        self.is_expected_call(llvm_link_name)
    }

    pub fn is_target_feature_call(&self) -> bool {
        self.is_expected_call("target_feature")
    }

    pub fn is_expected_call(&self, fn_call_name: &str) -> bool {
        if let Expression::Identifier(fn_name, IdentifierType::Symbol) = self.0.as_ref() {
            &fn_name.to_string() == fn_call_name
        } else {
            false
        }
    }

    pub fn pre_build(&mut self, ctx: &mut Context) -> context::Result {
        self.0.pre_build(ctx)?;
        self.1
            .iter_mut()
            .chain(self.2.iter_mut())
            .try_for_each(|ex| ex.pre_build(ctx))
    }

    pub fn build(&mut self, intrinsic: &Intrinsic, ctx: &mut Context) -> context::Result {
        self.0.build(intrinsic, ctx)?;
        self.1
            .iter_mut()
            .chain(self.2.iter_mut())
            .try_for_each(|ex| ex.build(intrinsic, ctx))
    }
}

impl ToTokens for FnCall {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let FnCall(fn_ptr, arguments, turbofish, _requires_unsafe_wrapper) = self;

        fn_ptr.to_tokens(tokens);

        if !turbofish.is_empty() {
            tokens.append_all(quote! {::<#(#turbofish),*>});
        }

        tokens.append_all(quote! { (#(#arguments),*) })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub enum Expression {
    /// (Re)Defines a variable
    Let(LetVariant),
    /// Performs a variable assignment operation
    Assign(String, Box<Expression>),
    /// Performs a macro call
    MacroCall(String, String),
    /// Performs a function call
    FnCall(FnCall),
    /// Performs a method call. The following:
    /// `MethodCall: ["$object", "to_string", []]`
    /// is tokenized as:
    /// `object.to_string()`.
    MethodCall(Box<Expression>, String, Vec<Expression>),
    /// Symbol identifier name, prepend with a `$` to treat it as a scope variable
    /// which engages variable tracking and enables inference.
    /// E.g. `my_function_name` for a generic symbol or `$my_variable` for
    /// a variable.
    Identifier(WildString, IdentifierType),
    /// Constant signed integer number expression
    IntConstant(i32),
    /// Constant floating point number expression
    FloatConstant(f32),
    /// Constant boolean expression, either `true` or `false`
    BoolConstant(bool),
    /// Array expression
    Array(Vec<Expression>),

    // complex expressions
    /// Makes an LLVM link.
    ///
    /// It stores the link's function name in the wildcard `{llvm_link}`, for use in
    /// subsequent expressions.
    LLVMLink(LLVMLink),
    /// Casts the given expression to the specified (unchecked) type
    CastAs(Box<Expression>, String),
    /// Returns the LLVM `undef` symbol
    SvUndef,
    /// Multiplication
    Multiply(Box<Expression>, Box<Expression>),
    /// Xor
    Xor(Box<Expression>, Box<Expression>),
    /// Converts the specified constant to the specified type's kind
    ConvertConst(TypeKind, i32),
    /// Yields the given type in the Rust representation
    Type(TypeKind),

    MatchSize(TypeKind, MatchSizeValues<Box<Expression>>),
    MatchKind(TypeKind, MatchKindValues<Box<Expression>>),
}

impl Expression {
    pub fn pre_build(&mut self, ctx: &mut Context) -> context::Result {
        match self {
            Self::FnCall(fn_call) => fn_call.pre_build(ctx),
            Self::MethodCall(cl_ptr_ex, _, arg_exs) => {
                cl_ptr_ex.pre_build(ctx)?;
                arg_exs.iter_mut().try_for_each(|ex| ex.pre_build(ctx))
            }
            Self::Let(
                LetVariant::Basic(_, ex)
                | LetVariant::WithType(_, _, ex)
                | LetVariant::MutWithType(_, _, ex),
            ) => ex.pre_build(ctx),
            Self::CastAs(ex, _) => ex.pre_build(ctx),
            Self::Multiply(lhs, rhs) | Self::Xor(lhs, rhs) => {
                lhs.pre_build(ctx)?;
                rhs.pre_build(ctx)
            }
            Self::MatchSize(match_ty, values) => {
                *self = *values.get(match_ty, ctx.local)?.to_owned();
                self.pre_build(ctx)
            }
            Self::MatchKind(match_ty, values) => {
                *self = *values.get(match_ty, ctx.local)?.to_owned();
                self.pre_build(ctx)
            }
            _ => Ok(()),
        }
    }

    pub fn build(&mut self, intrinsic: &Intrinsic, ctx: &mut Context) -> context::Result {
        match self {
            Self::LLVMLink(link) => link.build_and_save(ctx),
            Self::Identifier(identifier, id_type) => {
                identifier.build_acle(ctx.local)?;

                if let IdentifierType::Variable = id_type {
                    ctx.local
                        .variables
                        .get(&identifier.to_string())
                        .map(|_| ())
                        .ok_or_else(|| format!("invalid variable {identifier} being referenced"))
                } else {
                    Ok(())
                }
            }
            Self::FnCall(fn_call) => {
                fn_call.build(intrinsic, ctx)?;

                if let Some(llvm_link_name) = ctx.local.substitutions.get(&Wildcard::LLVMLink) {
                    if fn_call.is_llvm_link_call(llvm_link_name) {
                        *self = intrinsic
                            .llvm_link()
                            .expect("got LLVMLink wildcard without a LLVM link in `compose`")
                            .apply_conversions_to_call(fn_call.clone(), ctx)?
                    }
                }

                Ok(())
            }
            Self::MethodCall(cl_ptr_ex, _, arg_exs) => {
                cl_ptr_ex.build(intrinsic, ctx)?;
                arg_exs
                    .iter_mut()
                    .try_for_each(|ex| ex.build(intrinsic, ctx))
            }
            Self::Let(variant) => {
                let (var_name, ex, ty) = match variant {
                    LetVariant::Basic(var_name, ex) => (var_name, ex, None),
                    LetVariant::WithType(var_name, ty, ex)
                    | LetVariant::MutWithType(var_name, ty, ex) => {
                        if let Some(w) = ty.wildcard() {
                            ty.populate_wildcard(ctx.local.provide_type_wildcard(w)?)?;
                        }
                        (var_name, ex, Some(ty.to_owned()))
                    }
                };

                var_name.build_acle(ctx.local)?;
                ctx.local.variables.insert(
                    var_name.to_string(),
                    (
                        ty.unwrap_or_else(|| TypeKind::Custom("unknown".to_string())),
                        VariableType::Internal,
                    ),
                );
                ex.build(intrinsic, ctx)
            }
            Self::CastAs(ex, _) => ex.build(intrinsic, ctx),
            Self::Multiply(lhs, rhs) | Self::Xor(lhs, rhs) => {
                lhs.build(intrinsic, ctx)?;
                rhs.build(intrinsic, ctx)
            }
            Self::ConvertConst(ty, num) => {
                if let Some(w) = ty.wildcard() {
                    *ty = ctx.local.provide_type_wildcard(w)?
                }

                if let Some(BaseType::Sized(BaseTypeKind::Float, _)) = ty.base() {
                    *self = Expression::FloatConstant(*num as f32)
                } else {
                    *self = Expression::IntConstant(*num)
                }
                Ok(())
            }
            Self::Type(ty) => {
                if let Some(w) = ty.wildcard() {
                    *ty = ctx.local.provide_type_wildcard(w)?
                }

                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// True if the expression requires an `unsafe` context in a safe function.
    ///
    /// The classification is somewhat fuzzy, based on actual usage (e.g. empirical function names)
    /// rather than a full parse. This is a reasonable approach because mistakes here will usually
    /// be caught at build time:
    ///
    ///  - Missing an `unsafe` is a build error.
    ///  - An unnecessary `unsafe` is a warning, made into an error by the CI's `-D warnings`.
    ///
    /// This **panics** if it encounters an expression that shouldn't appear in a safe function at
    /// all (such as `SvUndef`).
    pub fn requires_unsafe_wrapper(&self, ctx_fn: &str) -> bool {
        match self {
            // The call will need to be unsafe, but the declaration does not.
            Self::LLVMLink(..) => false,
            // Identifiers, literals and type names are never unsafe.
            Self::Identifier(..) => false,
            Self::IntConstant(..) => false,
            Self::FloatConstant(..) => false,
            Self::BoolConstant(..) => false,
            Self::Type(..) => false,
            Self::ConvertConst(..) => false,
            // Nested structures that aren't inherently unsafe, but could contain other expressions
            // that might be.
            Self::Assign(_var, exp) => exp.requires_unsafe_wrapper(ctx_fn),
            Self::Let(
                LetVariant::Basic(_, exp)
                | LetVariant::WithType(_, _, exp)
                | LetVariant::MutWithType(_, _, exp),
            ) => exp.requires_unsafe_wrapper(ctx_fn),
            Self::Array(exps) => exps.iter().any(|exp| exp.requires_unsafe_wrapper(ctx_fn)),
            Self::Multiply(lhs, rhs) | Self::Xor(lhs, rhs) => {
                lhs.requires_unsafe_wrapper(ctx_fn) || rhs.requires_unsafe_wrapper(ctx_fn)
            }
            Self::CastAs(exp, _ty) => exp.requires_unsafe_wrapper(ctx_fn),
            // Functions and macros can be unsafe, but can also contain other expressions.
            Self::FnCall(FnCall(fn_exp, args, turbo_args, requires_unsafe_wrapper)) => {
                let fn_name = fn_exp.to_string();
                fn_exp.requires_unsafe_wrapper(ctx_fn)
                    || fn_name.starts_with("_sv")
                    || fn_name.starts_with("simd_")
                    || fn_name.ends_with("transmute")
                    || args.iter().any(|exp| exp.requires_unsafe_wrapper(ctx_fn))
                    || turbo_args
                        .iter()
                        .any(|exp| exp.requires_unsafe_wrapper(ctx_fn))
                    || *requires_unsafe_wrapper
            }
            Self::MethodCall(exp, fn_name, args) => match fn_name.as_str() {
                // `as_signed` and `as_unsigned` are unsafe because they're trait methods with
                // target features to allow use on feature-dependent types (such as SVE vectors).
                // We can safely wrap them here.
                "as_signed" => true,
                "as_unsigned" => true,
                _ => {
                    exp.requires_unsafe_wrapper(ctx_fn)
                        || args.iter().any(|exp| exp.requires_unsafe_wrapper(ctx_fn))
                }
            },
            // We only use macros to check const generics (using static assertions).
            Self::MacroCall(_name, _args) => false,
            // Materialising uninitialised values is always unsafe, and we avoid it in safe
            // functions.
            Self::SvUndef => panic!("Refusing to wrap unsafe SvUndef in safe function '{ctx_fn}'."),
            // Variants that aren't tokenised. We shouldn't encounter these here.
            Self::MatchKind(..) => {
                unimplemented!("The unsafety of {self:?} cannot be determined in '{ctx_fn}'.")
            }
            Self::MatchSize(..) => {
                unimplemented!("The unsafety of {self:?} cannot be determined in '{ctx_fn}'.")
            }
        }
    }

    /// Determine if an expression is a `static_assert<...>` function call.
    pub fn is_static_assert(&self) -> bool {
        match self {
            Expression::FnCall(fn_call) => match fn_call.0.as_ref() {
                Expression::Identifier(wild_string, _) => {
                    if let WildStringPart::String(function_name) = &wild_string.0[0] {
                        function_name.starts_with("static_assert")
                    } else {
                        false
                    }
                }
                _ => panic!("Badly defined function call: {:?}", fn_call),
            },
            _ => false,
        }
    }

    /// Determine if an espression is a LLVM binding
    pub fn is_llvm_link(&self) -> bool {
        if let Expression::LLVMLink(_) = self {
            true
        } else {
            false
        }
    }
}

impl FromStr for Expression {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        lazy_static! {
            static ref MACRO_RE: Regex =
                Regex::new(r"^(?P<name>[\w\d_]+)!\((?P<ex>.*?)\);?$").unwrap();
        }

        if s == "SvUndef" {
            Ok(Expression::SvUndef)
        } else if MACRO_RE.is_match(s) {
            let c = MACRO_RE.captures(s).unwrap();
            let ex = c["ex"].to_string();
            let _: TokenStream = ex
                .parse()
                .map_err(|e| format!("could not parse macro call expression: {e:#?}"))?;
            Ok(Expression::MacroCall(c["name"].to_string(), ex))
        } else {
            let (s, id_type) = if let Some(varname) = s.strip_prefix('$') {
                (varname, IdentifierType::Variable)
            } else {
                (s, IdentifierType::Symbol)
            };
            let identifier = s.trim().parse()?;
            Ok(Expression::Identifier(identifier, id_type))
        }
    }
}

impl From<FnCall> for Expression {
    fn from(fn_call: FnCall) -> Self {
        Expression::FnCall(fn_call)
    }
}

impl From<WildString> for Expression {
    fn from(ws: WildString) -> Self {
        Expression::Identifier(ws, IdentifierType::Symbol)
    }
}

impl From<&Argument> for Expression {
    fn from(a: &Argument) -> Self {
        Expression::Identifier(a.name.to_owned(), IdentifierType::Variable)
    }
}

impl TryFrom<&StaticDefinition> for Expression {
    type Error = String;

    fn try_from(sd: &StaticDefinition) -> Result<Self, Self::Error> {
        match sd {
            StaticDefinition::Constant(imm) => Ok(imm.into()),
            StaticDefinition::Generic(t) => t.parse(),
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identifier(identifier, kind) => {
                write!(
                    f,
                    "{}{identifier}",
                    matches!(kind, IdentifierType::Variable)
                        .then_some("$")
                        .unwrap_or_default()
                )
            }
            Self::MacroCall(name, expression) => {
                write!(f, "{name}!({expression})")
            }
            _ => Err(fmt::Error),
        }
    }
}

impl ToTokens for Expression {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Let(LetVariant::Basic(var_name, exp)) => {
                let var_ident = format_ident!("{}", var_name.to_string());
                tokens.append_all(quote! { let #var_ident = #exp })
            }
            Self::Let(LetVariant::WithType(var_name, ty, exp)) => {
                let var_ident = format_ident!("{}", var_name.to_string());
                tokens.append_all(quote! { let #var_ident: #ty = #exp })
            }
            Self::Let(LetVariant::MutWithType(var_name, ty, exp)) => {
                let var_ident = format_ident!("{}", var_name.to_string());
                tokens.append_all(quote! { let mut #var_ident: #ty = #exp })
            }
            Self::Assign(var_name, exp) => {
                /* If we are dereferencing a variable to assign a value \
                 * the 'format_ident!' macro does not like the asterix */
                let var_name_str: &str;

                if let Some(ch) = var_name.chars().nth(0) {
                    /* Manually append the asterix and split out the rest of
                     * the variable name */
                    if ch == '*' {
                        tokens.append(Punct::new('*', Spacing::Alone));
                        var_name_str = &var_name[1..var_name.len()];
                    } else {
                        var_name_str = var_name.as_str();
                    }
                } else {
                    /* Should not be reached as you cannot have a variable
                     * without a name */
                    panic!("Invalid variable name, must be at least one character")
                }

                let var_ident = format_ident!("{}", var_name_str);
                tokens.append_all(quote! { #var_ident = #exp })
            }
            Self::MacroCall(name, ex) => {
                let name = format_ident!("{name}");
                let ex: TokenStream = ex.parse().unwrap();
                tokens.append_all(quote! { #name!(#ex) })
            }
            Self::FnCall(fn_call) => fn_call.to_tokens(tokens),
            Self::MethodCall(exp, fn_name, args) => {
                let fn_ident = format_ident!("{}", fn_name);
                tokens.append_all(quote! { #exp.#fn_ident(#(#args),*) })
            }
            Self::Identifier(identifier, _) => {
                assert!(
                    !identifier.has_wildcards(),
                    "expression {self:#?} was not built before calling to_tokens"
                );
                identifier
                    .to_string()
                    .parse::<TokenStream>()
                    .expect(format!("invalid syntax: {:?}", self).as_str())
                    .to_tokens(tokens);
            }
            Self::IntConstant(n) => tokens.append(Literal::i32_unsuffixed(*n)),
            Self::FloatConstant(n) => tokens.append(Literal::f32_unsuffixed(*n)),
            Self::BoolConstant(true) => tokens.append(format_ident!("true")),
            Self::BoolConstant(false) => tokens.append(format_ident!("false")),
            Self::Array(vec) => tokens.append_all(quote! { [ #(#vec),* ] }),
            Self::LLVMLink(link) => link.to_tokens(tokens),
            Self::CastAs(ex, ty) => {
                let ty: TokenStream = ty.parse().expect("invalid syntax");
                tokens.append_all(quote! { #ex as #ty })
            }
            Self::SvUndef => tokens.append_all(quote! { simd_reinterpret(()) }),
            Self::Multiply(lhs, rhs) => tokens.append_all(quote! { #lhs * #rhs }),
            Self::Xor(lhs, rhs) => tokens.append_all(quote! { #lhs ^ #rhs }),
            Self::Type(ty) => ty.to_tokens(tokens),
            _ => unreachable!("{self:?} cannot be converted to tokens."),
        }
    }
}

impl Serialize for Expression {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::IntConstant(v) => serializer.serialize_i32(*v),
            Self::FloatConstant(v) => serializer.serialize_f32(*v),
            Self::BoolConstant(v) => serializer.serialize_bool(*v),
            Self::Identifier(..) => serializer.serialize_str(&self.to_string()),
            Self::MacroCall(..) => serializer.serialize_str(&self.to_string()),
            _ => Expression::serialize(self, serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Expression {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct CustomExpressionVisitor;

        impl<'de> Visitor<'de> for CustomExpressionVisitor {
            type Value = Expression;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("integer, float, boolean, string or map")
            }

            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Expression::BoolConstant(v))
            }

            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Expression::IntConstant(v as i32))
            }

            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Expression::IntConstant(v as i32))
            }

            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Expression::FloatConstant(v as f32))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                FromStr::from_str(value).map_err(de::Error::custom)
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let arr = std::iter::from_fn(|| seq.next_element::<Self::Value>().transpose())
                    .try_collect()?;
                Ok(Expression::Array(arr))
            }

            fn visit_map<M>(self, map: M) -> Result<Expression, M::Error>
            where
                M: MapAccess<'de>,
            {
                // `MapAccessDeserializer` is a wrapper that turns a `MapAccess`
                // into a `Deserializer`, allowing it to be used as the input to T's
                // `Deserialize` implementation. T then deserializes itself using
                // the entries from the map visitor.
                Expression::deserialize(de::value::MapAccessDeserializer::new(map))
            }
        }

        deserializer.deserialize_any(CustomExpressionVisitor)
    }
}
