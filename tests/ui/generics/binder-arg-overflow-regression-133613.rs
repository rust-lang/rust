//! Regression test for #133613
//! This was a crash regarding bound vars and generic arg indexing.
//! The bound var index was used to index self.args, and caused OOB.
//! Because of new changes to how the compiler handles binders, this was fixed.

//@ needs-rustc-debug-assertions

struct Wrapper<'a>();

// FIXME: these errors are probably wrong
trait IntFactory {
    fn stream(&self) -> impl IntFactory<stream(..): IntFactory<stream(..): Send>>;
    //~^ ERROR: cycle detected when looking up late bound vars inside `IntFactory::stream` [E0391]
    //~^^ ERROR: return type notation is experimental
    //~^^^ ERROR: return type notation is experimental
}

fn main() {}
