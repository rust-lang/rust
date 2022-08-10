// Regression test for #100267
//
// Previously the hidden type for Fut was `call::<'a, 'empty>::closure#0`,
// which failed WF checks cecause of the bound `'b: 'a`.
// Now we infer it to be `call::<'a, 'a>::closure#0`.
//
// Note that this is a pesky hack to workaround #100372.

// check-pass

#![feature(type_alias_impl_trait)]

type Fut<'a> = impl Sized;

fn call<'a, 'b>() -> Fut<'a>
where
    'b: 'a,
{
    || {}
}

fn main() {}
