#![feature(trait_alias)]

trait Svc<Req> { type Res; }

trait MkSvc<Target, Req> = Svc<Target> where Self::Res: Svc<Req>;
//~^ ERROR associated type `Res` not found for `Self`

fn main() {}
