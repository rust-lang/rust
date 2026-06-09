// This is a non-regression test for issue #114325: an "unexpected unsized tail" ICE happened during
// codegen, and was fixed by MIR drop tracking #107421.

//@ edition: 2021
//@ build-pass: ICEd during codegen.

#![feature(impl_trait_in_assoc_type)]

use std::future::Future;

fn main() {
    RuntimeRef::spawn_local(actor_fn(http_actor));
}

async fn http_actor() {
    async fn respond(body: impl Body) {
        body.write_message().await;
    }

    respond(&()).await;
}

trait Body {
    type WriteFuture: Future;

    fn write_message(self) -> Self::WriteFuture;
}

impl Body for &'static () {
    type WriteFuture = impl Future<Output = ()>;

    fn write_message(self) -> Self::WriteFuture {
        async {}
    }
}

trait NewActor {
    type RuntimeAccess;
}

fn actor_fn<T, A>(_d: T) -> (T, A) {
    loop {}
}

impl<F: FnMut() -> A, A> NewActor for (F, A) {
    type RuntimeAccess = RuntimeRef;
}
struct RuntimeRef(Vec<()>);

impl RuntimeRef {
    fn spawn_local<NA: NewActor<RuntimeAccess = RuntimeRef>>(_f: NA) {
        struct ActorFuture<NA: NewActor>(NA::RuntimeAccess);
        (ActorFuture::<NA>(RuntimeRef(vec![])), _f);
    }
}
