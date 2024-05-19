//@ edition:2018

async fn foo () { }
fn bar() -> impl std::future::Future { async {} }
fn boo() {}

async fn baz() -> std::io::Result<()> {
    foo().await;
    boo().await; //~ ERROR `()` is not a future
    bar().await;
    std::io::Result::Ok(())
}

macro_rules! e {
    () => {
        ()
    };
}

macro_rules! f {
    ($expr:expr) => {
        $expr.await
        //~^ ERROR `()` is not a future
    };
}

async fn with_macros() {
    e!().await;
    //~^ ERROR `()` is not a future

    f!(());
}

// Regression test for issue #117014.
async fn desugaring_span_ctxt() {
    for x in [] {}.await //~ ERROR `()` is not a future
}

fn main() {}
