use rustc_ast::MetaItemInner;
use rustc_attr_parsing as attr;
use rustc_middle::ty::{self, TyCtxt};
use rustc_parse_format::{ParseMode, Parser, Piece, Position};
use rustc_span::{Span, Symbol, kw, sym};

pub static ALLOWED_CONDITION_SYMBOLS: &[Symbol] = &[
    sym::from_desugaring,
    sym::direct,
    sym::cause,
    sym::integral,
    sym::integer_,
    sym::float,
    sym::_Self,
    sym::crate_local,
];

#[derive(Debug)]
pub struct Condition {
    pub inner: MetaItemInner,
}

impl Condition {
    pub fn span(&self) -> Span {
        self.inner.span()
    }

    pub fn matches_predicate<'tcx>(&self, tcx: TyCtxt<'tcx>, options: &ConditionOptions) -> bool {
        attr::eval_condition(&self.inner, tcx.sess, Some(tcx.features()), &mut |cfg| {
            let value = cfg.value.map(|v| {
                // `with_no_visible_paths` is also used when generating the options,
                // so we need to match it here.
                ty::print::with_no_visible_paths!({
                    let mut parser = Parser::new(v.as_str(), None, None, false, ParseMode::Format);
                    let constructed_message = (&mut parser)
                        .map(|p| match p {
                            Piece::Lit(s) => s.to_owned(),
                            Piece::NextArgument(a) => match a.position {
                                Position::ArgumentNamed(arg) => {
                                    let s = Symbol::intern(arg);
                                    match options.generic_args.iter().find(|(k, _)| *k == s) {
                                        Some((_, val)) => val.to_string(),
                                        None => format!("{{{arg}}}"),
                                    }
                                }
                                Position::ArgumentImplicitlyIs(_) => String::from("{}"),
                                Position::ArgumentIs(idx) => format!("{{{idx}}}"),
                            },
                        })
                        .collect();
                    constructed_message
                })
            });

            options.contains(cfg.name, &value)
        })
    }
}

#[derive(Debug)]
pub struct ConditionOptions {
    pub self_types: Vec<String>,
    pub from_desugaring: Option<String>,
    pub cause: Option<String>,
    pub crate_local: bool,
    pub direct: bool,
    pub generic_args: Vec<(Symbol, String)>,
}

impl ConditionOptions {
    pub fn contains(&self, key: Symbol, value: &Option<String>) -> bool {
        match (key, value) {
            (sym::_Self | kw::SelfUpper, Some(value)) => self.self_types.contains(&value),
            // from_desugaring as a flag
            (sym::from_desugaring, None) => self.from_desugaring.is_some(),
            // from_desugaring as key == value
            (sym::from_desugaring, v) => *v == self.from_desugaring,
            (sym::cause, Some(value)) => self.cause.as_deref() == Some(value),
            (sym::crate_local, None) => self.crate_local,
            (sym::direct, None) => self.direct,
            (other, Some(value)) => {
                self.generic_args.iter().any(|(k, v)| *k == other && v == value)
            }
            _ => false,
        }
    }
}
