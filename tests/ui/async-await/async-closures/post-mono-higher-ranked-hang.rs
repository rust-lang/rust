//@ build-fail
//@ aux-build:block-on.rs
//@ edition:2021

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/135780>.

extern crate block_on;

use std::future::Future;
use std::ops::AsyncFnMut;
use std::pin::{Pin, pin};
use std::task::*;

trait Db {}

impl Db for () {}

struct Env<'db> {
    db: &'db (),
}

#[derive(Debug)]
enum SymPerm<'db> {
    Dummy(&'db ()),
    Apply(Box<SymPerm<'db>>, Box<SymPerm<'db>>),
}

pub struct ToChain<'env, 'db> {
    db: &'db dyn crate::Db,
    env: &'env Env<'db>,
}

impl<'env, 'db> ToChain<'env, 'db> {
    fn perm_pairs<'l>(
        &'l self,
        perm: &'l SymPerm<'db>,
        yield_chain: &'l mut impl AsyncFnMut(&SymPerm<'db>),
    ) -> Pin<Box<dyn std::future::Future<Output = ()> + 'l>> {
        Box::pin(async move {
            match perm {
                SymPerm::Dummy(_) => yield_chain(perm).await,
                SymPerm::Apply(l, r) => {
                    self.perm_pairs(l, &mut async move |left_pair| {
                        //~^ ERROR reached the recursion limit while instantiating
                        self.perm_pairs(r, yield_chain).await
                    })
                    .await
                }
            }
        })
    }
}

fn main() {
    block_on::block_on(async {
        let pair = SymPerm::Apply(Box::new(SymPerm::Dummy(&())), Box::new(SymPerm::Dummy(&())));
        ToChain { db: &(), env: &Env { db: &() } }
            .perm_pairs(&pair, &mut async |p| {
                eprintln!("{p:?}");
            })
            .await;
    });
}
