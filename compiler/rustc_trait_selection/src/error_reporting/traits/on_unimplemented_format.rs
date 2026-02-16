use std::fmt;

use rustc_hir::attrs::diagnostic::*;
use rustc_middle::ty::print::TraitRefPrintSugared;
use rustc_span::{Symbol, kw};

/// Arguments to fill a [FormatString] with.
///
/// For example, given a
/// ```rust,ignore (just an example)
///
/// #[rustc_on_unimplemented(
///     on(all(from_desugaring = "QuestionMark"),
///         message = "the `?` operator can only be used in {ItemContext} \
///                     that returns `Result` or `Option` \
///                     (or another type that implements `{FromResidual}`)",
///         label = "cannot use the `?` operator in {ItemContext} that returns `{Self}`",
///         parent_label = "this function should return `Result` or `Option` to accept `?`"
///     ),
/// )]
/// pub trait FromResidual<R = <Self as Try>::Residual> {
///    ...
/// }
///
/// async fn an_async_function() -> u32 {
///     let x: Option<u32> = None;
///     x?; //~ ERROR the `?` operator
///     22
/// }
///  ```
/// it will look like this:
///
/// ```rust,ignore (just an example)
/// FormatArgs {
///     this: "FromResidual",
///     trait_sugared: "FromResidual<Option<Infallible>>",
///     item_context: "an async function",
///     generic_args: [("Self", "u32"), ("R", "Option<Infallible>")],
/// }
/// ```
#[derive(Debug)]
pub struct FormatArgs<'tcx> {
    pub this: String,
    pub trait_sugared: TraitRefPrintSugared<'tcx>,
    pub item_context: &'static str,
    pub generic_args: Vec<(Symbol, String)>,
}

pub fn format(slf: &FormatString, args: &FormatArgs<'_>) -> String {
    let mut ret = String::new();
    for piece in &slf.pieces {
        match piece {
            Piece::Lit(s) | Piece::Arg(FormatArg::AsIs(s)) => ret.push_str(s.as_str()),

            // `A` if we have `trait Trait<A> {}` and `note = "i'm the actual type of {A}"`
            Piece::Arg(FormatArg::GenericParam { generic_param, .. }) => {
                match args.generic_args.iter().find(|(p, _)| p == generic_param) {
                    Some((_, val)) => ret.push_str(val.as_str()),

                    None => {
                        // Apparently this was not actually a generic parameter, so lets write
                        // what the user wrote.
                        let _ = fmt::write(&mut ret, format_args!("{{{generic_param}}}"));
                    }
                }
            }
            // `{Self}`
            Piece::Arg(FormatArg::SelfUpper) => {
                let slf = match args.generic_args.iter().find(|(p, _)| *p == kw::SelfUpper) {
                    Some((_, val)) => val.to_string(),
                    None => "Self".to_string(),
                };
                ret.push_str(&slf);
            }

            // It's only `rustc_onunimplemented` from here
            Piece::Arg(FormatArg::This) => ret.push_str(&args.this),
            Piece::Arg(FormatArg::Trait) => {
                let _ = fmt::write(&mut ret, format_args!("{}", &args.trait_sugared));
            }
            Piece::Arg(FormatArg::ItemContext) => ret.push_str(args.item_context),
        }
    }
    ret
}
