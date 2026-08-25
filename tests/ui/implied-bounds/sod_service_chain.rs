// Found in a crater run on #118553

//@ dont-require-annotations: NOTE

pub trait Debug {}

pub trait Service {
    type Input;
    type Output;
    type Error;
}

pub struct ServiceChain<P, S> {
    prev: P,
    service: S,
}
impl<P: Service, S: Service<Input = P::Output>> Service for ServiceChain<P, S>
where
    P::Error: 'static,
    S::Error: 'static,
{
    type Input = P::Input;
    type Output = S::Output;
    type Error = ();
}

pub struct ServiceChainBuilder<P: Service, S: Service<Input = P::Output>> {
    chain: ServiceChain<P, S>,
}
impl<P: Service, S: Service<Input = P::Output>> ServiceChainBuilder<P, S> {
    pub fn next<NS: Service<Input = S::Output>>(
        //~^ ERROR the associated type
        //~| ERROR the associated type
        //~| ERROR the associated type
        //~| NOTE the associated type
        //~| NOTE the associated type
        //~| NOTE the associated type
        //~| ERROR may not live long enough
        self,
    ) -> ServiceChainBuilder<ServiceChain<P, S>, NS> {
        //~^ ERROR the associated type
        //~| ERROR the associated type
        //~| NOTE the associated type
        //~| NOTE the associated type
        panic!();
    }
}

fn main() {}
