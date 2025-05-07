// Regression test for #135668
//@ edition: 2021

use std::future::Future;

pub async fn foo() {
    let _ = create_task().await;
}

async fn create_task() -> impl Sized {
    bind(documentation)
}

async fn documentation() {
    compile_error!("bonjour");
    //~^ ERROR bonjour
}

fn bind<F>(_filter: F) -> impl Sized
where
    F: FilterBase,
{
    || -> <F as FilterBase>::Assoc { panic!() }
}

trait FilterBase {
    type Assoc;
}

impl<F, R> FilterBase for F
where
    F: Fn() -> R,
    // Removing the below line makes it correctly error on both stable and beta
    R: Future,
    // Removing the below line makes it ICE on both stable and beta
    R: Send,
    // Removing the above two bounds makes it ICE on stable but correctly error on beta
{
    type Assoc = F;
}

fn main() {}
