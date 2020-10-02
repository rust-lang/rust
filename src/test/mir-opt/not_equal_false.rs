// EMIT_MIR not_equal_false.opt.InstCombine.diff

fn opt(x: Option<()>) -> bool {
    matches!(x, None) || matches!(x, Some(_))
}

fn main() {
    opt(None);
}
