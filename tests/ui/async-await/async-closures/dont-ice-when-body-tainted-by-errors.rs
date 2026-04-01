//@ edition: 2021

struct DropMe;

trait Impossible {}
fn trait_error<T: Impossible>() {}

pub fn main() {
    let b = DropMe;
    let async_closure = async move || {
        // Type error here taints the environment. This causes us to fallback all
        // variables to `Error`. This means that when we compute the upvars for the
        // *outer* coroutine-closure, we don't actually see any upvars since `MemCategorization`
        // and `ExprUseVisitor`` will bail early when it sees error. This means
        // that our underlying assumption that the parent and child captures are
        // compatible ends up being broken, previously leading to an ICE.
        trait_error::<()>();
        //~^ ERROR the trait bound `(): Impossible` is not satisfied
        let _b = b;
    };
}
