use rustc_hir::attrs::diagnostic::*;
use rustc_span::Symbol;

pub fn matches_predicate(slf: &OnUnimplementedCondition, options: &ConditionOptions) -> bool {
    slf.pred.eval(&mut |p| match p {
        FlagOrNv::Flag(b) => options.has_flag(*b),
        FlagOrNv::NameValue(NameValue { name, value }) => {
            let value = format_filter(value, &options.generic_args);
            options.contains(*name, value)
        }
    })
}

fn format_filter(slf: &FilterFormatString, generic_args: &[(Symbol, String)]) -> String {
    let mut ret = String::new();

    for piece in &slf.pieces {
        match piece {
            LitOrArg::Lit(s) => ret.push_str(s.as_str()),
            LitOrArg::Arg(s) => match generic_args.iter().find(|(k, _)| k == s) {
                Some((_, val)) => ret.push_str(val),
                None => {
                    let _ = std::fmt::write(&mut ret, format_args!("{{{s}}}"));
                }
            },
        }
    }

    ret
}
