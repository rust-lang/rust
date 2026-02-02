use rustc_ast::{MetaItemInner, MetaItemKind, MetaItemLit};
use rustc_hir::attrs::diagnostic::*;
use rustc_parse_format::{ParseMode, Parser, Piece, Position};
use rustc_span::{Ident, Symbol, kw, sym};
use thin_vec::ThinVec;

use crate::errors::InvalidOnClause;

pub fn matches_predicate(slf: &OnUnimplementedCondition, options: &ConditionOptions) -> bool {
    slf.pred.eval(&mut |p| match p {
        FlagOrNv::Flag(b) => options.has_flag(*b),
        FlagOrNv::NameValue(NameValue { name, value }) => {
            let value = format_filter(value, &options.generic_args);
            options.contains(*name, value)
        }
    })
}

pub(crate) fn parse_condition(
    input: &MetaItemInner,
    generics: &[Symbol],
) -> Result<OnUnimplementedCondition, InvalidOnClause> {
    let span = input.span();
    let pred = parse_predicate(input, generics)?;
    Ok(OnUnimplementedCondition { span, pred })
}

fn parse_predicate(
    input: &MetaItemInner,
    generics: &[Symbol],
) -> Result<Predicate, InvalidOnClause> {
    let meta_item = match input {
        MetaItemInner::MetaItem(meta_item) => meta_item,
        MetaItemInner::Lit(lit) => {
            return Err(InvalidOnClause::UnsupportedLiteral { span: lit.span });
        }
    };

    let Some(predicate) = meta_item.ident() else {
        return Err(InvalidOnClause::ExpectedIdentifier {
            span: meta_item.path.span,
            path: meta_item.path.clone(),
        });
    };

    match meta_item.kind {
        MetaItemKind::List(ref mis) => match predicate.name {
            sym::any => Ok(Predicate::Any(parse_predicate_sequence(mis, generics)?)),
            sym::all => Ok(Predicate::All(parse_predicate_sequence(mis, generics)?)),
            sym::not => match &**mis {
                [one] => Ok(Predicate::Not(Box::new(parse_predicate(one, generics)?))),
                [first, .., last] => Err(InvalidOnClause::ExpectedOnePredInNot {
                    span: first.span().to(last.span()),
                }),
                [] => Err(InvalidOnClause::ExpectedOnePredInNot { span: meta_item.span }),
            },
            invalid_pred => {
                Err(InvalidOnClause::InvalidPredicate { span: predicate.span, invalid_pred })
            }
        },
        MetaItemKind::NameValue(MetaItemLit { symbol, .. }) => {
            let name = parse_name(predicate, generics)?;
            let value = parse_filter(symbol);
            let kv = NameValue { name, value };
            Ok(Predicate::Match(kv))
        }
        MetaItemKind::Word => {
            let flag = parse_flag(predicate)?;
            Ok(Predicate::Flag(flag))
        }
    }
}

fn parse_predicate_sequence(
    sequence: &[MetaItemInner],
    generics: &[Symbol],
) -> Result<ThinVec<Predicate>, InvalidOnClause> {
    sequence.iter().map(|item| parse_predicate(item, generics)).collect()
}

fn parse_flag(Ident { name, span }: Ident) -> Result<Flag, InvalidOnClause> {
    match name {
        sym::crate_local => Ok(Flag::CrateLocal),
        sym::direct => Ok(Flag::Direct),
        sym::from_desugaring => Ok(Flag::FromDesugaring),
        invalid_flag => Err(InvalidOnClause::InvalidFlag { invalid_flag, span }),
    }
}

fn parse_name(Ident { name, span }: Ident, generics: &[Symbol]) -> Result<Name, InvalidOnClause> {
    match name {
        kw::SelfUpper => Ok(Name::SelfUpper),
        sym::from_desugaring => Ok(Name::FromDesugaring),
        sym::cause => Ok(Name::Cause),
        generic if generics.contains(&generic) => Ok(Name::GenericArg(generic)),
        invalid_name => Err(InvalidOnClause::InvalidName { invalid_name, span }),
    }
}

fn parse_filter(input: Symbol) -> FilterFormatString {
    let pieces = Parser::new(input.as_str(), None, None, false, ParseMode::Diagnostic)
        .map(|p| match p {
            Piece::Lit(s) => LitOrArg::Lit(Symbol::intern(s)),
            // We just ignore formatspecs here
            Piece::NextArgument(a) => match a.position {
                // In `TypeErrCtxt::on_unimplemented_note` we substitute `"{integral}"` even
                // if the integer type has been resolved, to allow targeting all integers.
                // `"{integer}"` and `"{float}"` come from numerics that haven't been inferred yet,
                // from the `Display` impl of `InferTy` to be precise.
                //
                // Don't try to format these later!
                Position::ArgumentNamed(arg @ "integer" | arg @ "integral" | arg @ "float") => {
                    LitOrArg::Lit(Symbol::intern(&format!("{{{arg}}}")))
                }

                // FIXME(mejrs) We should check if these correspond to a generic of the trait.
                Position::ArgumentNamed(arg) => LitOrArg::Arg(Symbol::intern(arg)),

                // FIXME(mejrs) These should really be warnings/errors
                Position::ArgumentImplicitlyIs(_) => LitOrArg::Lit(sym::empty_braces),
                Position::ArgumentIs(idx) => LitOrArg::Lit(Symbol::intern(&format!("{{{idx}}}"))),
            },
        })
        .collect();
    FilterFormatString { pieces }
}

fn format_filter(slf: &FilterFormatString, generic_args: &[(Symbol, String)]) -> String {
    let mut ret = String::new();

    for piece in &slf.pieces {
        match piece {
            LitOrArg::Lit(s) => ret.push_str(s.as_str()),
            LitOrArg::Arg(s) => {
                match generic_args.iter().find(|(k, _)| k == s) {
                    Some((_, val)) => ret.push_str(val),
                    None => {
                        // FIXME(mejrs) If we start checking as mentioned in
                        // FilterFormatString::parse then this shouldn't happen
                        let _ = std::fmt::write(&mut ret, format_args!("{{{s}}}"));
                    }
                }
            }
        }
    }

    ret
}
