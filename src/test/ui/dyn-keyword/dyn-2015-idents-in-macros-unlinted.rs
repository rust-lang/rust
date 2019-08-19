// build-pass (FIXME(62277): could be check-pass?)

// Under the 2015 edition with the keyword_idents lint, `dyn` is
// not entirely acceptable as an identifier.
//
// We currently do not attempt to detect or fix uses of `dyn` as an
// identifier under a macro.

#![allow(non_camel_case_types)]
#![deny(keyword_idents)]

mod outer_mod {
    pub mod r#dyn {
        pub struct r#dyn;
    }
}

// Here we are illustrating that the current lint does not flag the
// occurrences of `dyn` in this macro definition; however, it
// certainly *could* (and it would be nice if it did), since these
// occurrences are not compatible with the 2018 edition's
// interpretation of `dyn` as a keyword.
macro_rules! defn_has_dyn_idents {
    () => { ::outer_mod::dyn::dyn }
}

struct X;
trait Trait { fn hello(&self) { }}
impl Trait for X { }

macro_rules! tt_trait {
    ($arg:tt) => { & $arg Trait }
}

macro_rules! id_trait {
    ($id:ident) => { & $id Trait }
}

fn main() {
    defn_has_dyn_idents!();

    // Here we are illustrating that the current lint does not flag
    // the occurrences of `dyn` in these macro invocations. It
    // definitely should *not* flag the one in `tt_trait`, since that
    // is expanding in a valid fashion to `&dyn Trait`.
    //
    // It is arguable whether it would be valid to flag the occurrence
    // in `id_trait`, since that macro specifies that it takes an
    // `ident` as its input.
    fn f_tt(x: &X) -> tt_trait!(dyn) { x }
    fn f_id(x: &X) -> id_trait!(dyn) { x }

    let x = X;
    f_tt(&x).hello();
    f_id(&x).hello();
}
