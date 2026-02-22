// This is a regression test for <https://github.com/rust-lang/rust/issues/152797>.
#![feature(final_associated_functions)]
#![feature(min_generic_const_args)]
#![expect(incomplete_features)]
trait Uwu {
    final type Ovo;
    //~^ error: `final` is only allowed on associated functions in traits
    final type const QwQ: ();
    //~^ error: `final` is only allowed on associated functions in traits
}

fn main() {}
