//@ check-pass

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

struct Foo<const N: usize>;

trait Trait {
    #[type_const]
    const ASSOC: usize;
}

type Arr<const N: usize> = [(); {{{ N }}}];
type Arr2<T> = [(); {{{ <T as Trait>::ASSOC }}}];
type Ty<const N: usize> = Foo<{{{ N }}}>;
type Ty2<T> = Foo<{{{ <T as Trait>::ASSOC }}}>;
struct Default<const N: usize, const M: usize = {{{ N }}}>;
struct Default2<T: Trait, const M: usize = {{{ <T as Trait>::ASSOC }}}>(T);

fn repeat<T: Trait, const N: usize>() {
    let _1 = [(); {{{ N }}}];
    let _2 = [(); {{{ <T as Trait>::ASSOC }}}];
}

fn main() {}
