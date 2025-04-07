//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// A variation of #135246 where the cyclic bounds are part of
// the impl instead of the impl associated item.

trait Trait<R>: Sized {
    type Proof: Trait<R, Proof = Self>;
}

// We need to use indirection here as we otherwise normalize
// `<L::Proof as Trait<R>>::Proof` before recursing into
// `R: Trait<R, Proof = <L::Proof as Trait<R>>::Proof>`.
trait Indir<L: Trait<R>, R>: Trait<R, Proof = <L::Proof as Trait<R>>::Proof> {}
impl<L, R> Indir<L, R> for R
where
    L: Trait<R>,
    R: Trait<R, Proof = <L::Proof as Trait<R>>::Proof>,
{}

impl<L, R> Trait<R> for L
where
    L: Trait<R>,
    R: Indir<L, R>,
 {
    type Proof = R;
}
fn transmute<L: Trait<R>, R>(r: L) -> <L::Proof as Trait<R>>::Proof { r }
fn main() {
    let s: String = transmute::<_, String>(vec![65_u8, 66, 67]);
    //~^ ERROR overflow evaluating the requirement `Vec<u8>: Trait<String>`
    //[next]~| ERROR overflow evaluating the requirement `<<Vec<u8> as Trait<String>>::Proof as Trait<String>>::Proof == _`
    //[next]~| ERROR overflow evaluating the requirement `<<Vec<u8> as Trait<String>>::Proof as Trait<String>>::Proof == String`
    //[next]~| ERROR overflow evaluating the requirement `<<Vec<u8> as Trait<String>>::Proof as Trait<String>>::Proof: Sized`
    //[next]~| ERROR overflow evaluating the requirement `<<Vec<u8> as Trait<String>>::Proof as Trait<String>>::Proof well-formed`
    //[next]~| ERROR overflow evaluating the requirement `<<Vec<u8> as Trait<String>>::Proof as Trait<String>>::Proof == _`
    println!("{}", s); // ABC
}
