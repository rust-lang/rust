// MIR opt test: verifica que chains de && sobre bools puros
// são folded em BitAnd, eliminando phi nodes.
//
// EMIT_MIR bool_chain_opt.eq.BoolChainOpt.diff

pub struct S {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
}

// EMIT_MIR bool_chain_opt.eq.BoolChainOpt.diff
pub fn eq(x: &S, y: &S) -> bool {
    x.a == y.a && x.b == y.b && x.c == y.c && x.d == y.d
}
