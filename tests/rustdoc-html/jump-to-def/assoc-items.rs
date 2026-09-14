//@ compile-flags: -Zunstable-options --generate-link-to-definition

//@ has 'src/assoc_items/assoc-items.rs.html'

trait Trait0 {
    fn fn0();
    fn fn1();
    fn fn2();
    fn fn3();
    const CT0: usize;
    const CT1: usize;
    type Ty0;
    type Ty1;
    type Ty2: Bound0;
    type Ty3: Bound0;
}

trait Bound0 {
    fn fn0();
    const CT0: usize;
}

fn expr<T: Trait0>() {
    //@ has - '//a[@href="#6"]' 'fn0'
    let _ = <T as Trait0>::fn0;     // Expr, AssocFn,    Resolved
    //@ has - '//a[@href="#7"]' 'fn1'
    let _ = T::fn1;                 // Expr, AssocFn,    TypeRelative

    //@ has - '//a[@href="#8"]' 'fn2'
    let _ = <T as Trait0>::fn2();   // Expr, AssocFn,    Resolved
    //@ has - '//a[@href="#9"]' 'fn3'
    let _ = T::fn3();               // Expr, AssocFn,    TypeRelative

    //@ has - '//a[@href="#10"]' 'CT0'
    let _ = <T as Trait0>::CT0;     // Expr, AssocConst, Resolved
    //@ has - '//a[@href="#11"]' 'CT1'
    let _ = <T>::CT1;               // Expr, AssocConst, TypeRelative

    //@ has - '//a[@href="#12"]' 'Ty0'
    let _: <T as Trait0>::Ty0;      // Expr, AssocTy,    Resolved
    // FIXME: Support this:
    //@ !has - '//a[@href="#13"]' 'Ty1'
    let _: T::Ty1;                  // Expr, AssocTy,    TypeRelative

    //@ has - '//a[@href="#14"]' 'Ty2'
    //@ has - '//a[@href="#19"]' 'fn0'
    let _ = <T as Trait0>::Ty2::fn0();

    // FIXME: Support this:
    //@ !has - '//a[@href="#14"]' 'Ty3'
    //@ has - '//a[@href="#20"]' 'CT0'
    let _ = T::Ty2::CT0;
}

trait Trait1 {
    const CT0: usize;
    const CT1: usize;
    const CT2: usize;

    fn scope();
}


fn pat() {
    //@ has - '//a[@href="#56"]' 'CT0'
    if let <() as Trait1>::CT0 = 0 {}   // Pat,  AssocConst, Resolved

    match 0 {
        //@ has - '//a[@href="#57"]' 'CT1'
        <() as Trait1>::CT1 => {}       // Pat,  AssocConst, Resolved
        _ => {}
    }
}

impl Trait1 for () {
    const CT0: usize = 0;
    const CT1: usize = 1;
    const CT2: usize = 2;

    fn scope() {
        //@ has - '//a[@href="#58"]' 'CT2'
        if let Self::CT2 = 0 {}         // Pat,  AssocConst, TypeRelative
    }
}

trait Trait2 {
    const CT0: usize;
    type Ty0;
    type Ty1;
}

impl Trait2 for () {
    const CT0: usize = 0;
    type Ty0 = ();
    type Ty1 = ();
}

struct Item<T: Trait2> {
    //@ has - '//a[@href="#87"]' 'CT0'
    f0: [(); <() as Trait2>::CT0],   // Item, AssocConst, Resolved
    //@ has - '//a[@href="#88"]' 'Ty0'
    f1: <T as Trait2>::Ty0,          // Item, AssocTy,    Resolved
    // FIXME: Support this:
    //@ !has - '//a[@href="#89"]' 'Ty1'
    f2: T::Ty1,                      // Item, AssocTy,    TypeRelative
}
