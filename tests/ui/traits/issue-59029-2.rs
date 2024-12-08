//@ check-pass
#![feature(trait_alias)]

trait Svc<Req> { type Res; }

trait MkSvc<Target, Req> = Svc<Target> where <Self as Svc<Target>>::Res: Svc<Req>;

fn main() {}
