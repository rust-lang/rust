//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ edition:2021

mod hyper {
    use std::{fmt::Debug, future::Future, marker::PhantomData, pin::Pin, task::Poll};

    pub trait HttpBody {
        type Error;
    }
    impl HttpBody for () {
        //~^ ERROR not all trait items implemented, missing: `Error`
        // don't implement `Error` here for the ICE
    }

    pub struct Server<I, S>(I, S);

    pub fn serve<I, S>(_: S) -> Server<I, S> {
        todo!()
    }

    impl<S, B> Future for Server<(), S>
    where
        S: MakeServiceRef<(), (), ResBody = B>,
        B: HttpBody,
        B::Error: Debug,
    {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _: &mut std::task::Context<'_>) -> Poll<Self::Output> {
            todo!()
        }
    }

    pub trait MakeServiceRef<Target, ReqBody> {
        type ResBody;
    }

    impl<T, S> MakeServiceRef<(), ()> for T
    where
        T: for<'a> Service<&'a (), Response = S>,
        S: Service<()>,
    {
        type ResBody = ();
    }

    pub struct MakeServiceFn<F>(pub F);
    pub struct ServiceFn<F, R>(pub PhantomData<(F, R)>);

    pub trait Service<Request> {
        type Response;
    }

    impl<'t, F, Ret, Target, Svc> Service<&'t Target> for MakeServiceFn<F>
    where
        F: Fn() -> Ret,
        Ret: Future<Output = Result<Svc, ()>>,
    {
        type Response = Svc;
    }

    impl<F, ReqBody, Ret, ResBody, E> Service<ReqBody> for ServiceFn<F, ReqBody>
    where
        F: Fn() -> Ret,
        Ret: Future<Output = Result<ResBody, E>>,
    {
        type Response = ResBody;
    }
}

async fn smarvice() -> Result<(), ()> {
    Ok(())
}

fn service_fn<F, R, S>(f: F) -> hyper::ServiceFn<F, R>
where
    F: Fn() -> S,
{
    hyper::ServiceFn(std::marker::PhantomData)
}

async fn iceice() {
    let service = hyper::MakeServiceFn(|| async { Ok::<_, ()>(service_fn(|| smarvice())) });
    hyper::serve::<(), _>(service).await;
}

fn main() {}
