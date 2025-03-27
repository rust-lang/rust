use rustc_ast::MetaItemInner;
use rustc_attr_parsing as attr;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::{self, TyCtxt};
use rustc_parse_format::{ParseMode, Parser, Piece, Position};
use rustc_span::{Span, Symbol, sym};

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

    pub fn matches_predicate<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        options: &[(Symbol, Option<String>)],
        options_map: &FxHashMap<Symbol, String>,
    ) -> bool {
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
                                    match options_map.get(&s) {
                                        Some(val) => val.to_string(),
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

            options.contains(&(cfg.name, value))
        })
    }
}
