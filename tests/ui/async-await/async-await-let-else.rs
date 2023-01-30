// edition:2021
// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir

use std::rc::Rc;

async fn foo(x: Option<bool>) {
    let Some(_) = x else {
        let r = Rc::new(());
        bar().await
    };
}

async fn bar() -> ! {
    panic!()
}

fn is_send<T: Send>(_: T) {}

async fn foo2(x: Option<bool>) {
    let Some(_) = x else {
        bar2(Rc::new(())).await
    };
}

async fn bar2<T>(_: T) -> ! {
    panic!()
}

async fn foo3(x: Option<bool>) {
    let Some(_) = x else {
        (Rc::new(()), bar().await);
        return;
    };
}

async fn foo4(x: Option<bool>) {
    let Some(_) = x else {
        let r = Rc::new(());
        bar().await;
        println!("{:?}", r);
        return;
    };
}

fn main() {
    is_send(foo(Some(true)));
    //~^ ERROR cannot be sent between threads safely
    is_send(foo2(Some(true)));
    //~^ ERROR cannot be sent between threads safely
    is_send(foo3(Some(true)));
    //~^ ERROR cannot be sent between threads safely
    is_send(foo4(Some(true)));
    //~^ ERROR cannot be sent between threads safely
}
