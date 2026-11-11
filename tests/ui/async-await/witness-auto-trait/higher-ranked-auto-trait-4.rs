// Repro for <https://github.com/rust-lang/rust/issues/102211#issuecomment-2891975128>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

use std::future::Future;

trait BoringTrait {}

trait TraitWithAssocType<I> {
    type Assoc;
}

impl<T> TraitWithAssocType<()> for T
where
    T: ?Sized + 'static,
{
    type Assoc = ();
}

fn evil_function<T: TraitWithAssocType<I> + ?Sized, I>()
-> impl Future<Output = Result<(), T::Assoc>> {
    async { Ok(()) }
}

fn fails_to_compile() -> impl std::future::Future<Output = ()> + Send {
    async {
        let _ = evil_function::<dyn BoringTrait, _>().await;
    }
}

fn main() {}
