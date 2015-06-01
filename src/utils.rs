use rustc::lint::Context;
use syntax::codemap::ExpnInfo;

fn in_macro(cx: &Context, opt_info: Option<&ExpnInfo>) -> bool {
	opt_info.map_or(false, |info| {
		info.callee.span.map_or(true, |span| {
			cx.sess().codemap().span_to_snippet(span).ok().map_or(true, |code| 
				!code.starts_with("macro_rules")
			)
		})
	})
}
