// Issue #72117
// edition:2018

use core::future::Future;
use core::pin::Pin;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

impl<T: ?Sized> FutureExt for T where T: Future {}
trait FutureExt: Future {
    fn boxed<'a>(self) -> BoxFuture<'a, Self::Output>
    where
        Self: Sized + Send + 'a,
    {
        Box::pin(self)
    }
}

fn main() {
    let _: BoxFuture<'static, bool> = async {}.boxed();
    //~^ ERROR: mismatched types
}
