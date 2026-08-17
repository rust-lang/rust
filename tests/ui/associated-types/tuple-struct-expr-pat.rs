// Check that fully qualified syntax can **not** be used in tuple struct expressions (calls) and
// patterns. Both tuple struct expressions and patterns are resolved in value namespace and thus
// can't be resolved through associated *types*.
//
//@ run-rustfix

#![feature(more_qualified_paths)]

fn main() {
    let <T<0> as Trait>::Assoc() = <T<0> as Trait>::Assoc();
    //~^ error: cannot find method or associated constant `Assoc` in trait `Trait`
    //~| error: cannot find tuple struct or tuple variant `Assoc` in trait `Trait`
    let <T<1> as Trait>::Assoc(_a) = <T<1> as Trait>::Assoc(0);
    //~^ error: cannot find method or associated constant `Assoc` in trait `Trait`
    //~| error: cannot find tuple struct or tuple variant `Assoc` in trait `Trait`
    let <T<2> as Trait>::Assoc(_a, _b) = <T<2> as Trait>::Assoc(0, 1);
    //~^ error: cannot find method or associated constant `Assoc` in trait `Trait`
    //~| error: cannot find tuple struct or tuple variant `Assoc` in trait `Trait`
    let <T<3> as Trait>::Assoc(ref _a, ref mut _b, mut _c) = <T<3> as Trait>::Assoc(0, 1, 2);
    //~^ error: cannot find method or associated constant `Assoc` in trait `Trait`
    //~| error: cannot find tuple struct or tuple variant `Assoc` in trait `Trait`
}


struct T<const N: usize>;

struct T0();
struct T1(u8);
struct T2(u8, u8);
struct T3(u8, u8, u8);

trait Trait {
    type Assoc;
}

impl Trait for T<0> {
    type Assoc = T0;
}

impl Trait for T<1> {
    type Assoc = T1;
}
impl Trait for T<2> {
    type Assoc = T2;
}
impl Trait for T<3> {
    type Assoc = T3;
}
