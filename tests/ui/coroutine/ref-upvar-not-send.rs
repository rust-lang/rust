// For `Send` coroutines, suggest a `T: Sync` requirement for `&T` upvars,
// and suggest a `T: Send` requirement for `&mut T` upvars.

#![feature(coroutines, stmt_expr_attributes)]

fn assert_send<T: Send>(_: T) {}
//~^ NOTE required by a bound in `assert_send`
//~| NOTE required by this bound in `assert_send`
//~| NOTE required by a bound in `assert_send`
//~| NOTE required by this bound in `assert_send`
//~| NOTE required by a bound in `assert_send`
//~| NOTE required by this bound in `assert_send`
//~| NOTE required by a bound in `assert_send`
//~| NOTE required by this bound in `assert_send`

fn main() {
    let x: &*mut () = &std::ptr::null_mut();
    //~^ NOTE has type `&*mut ()` which is not `Send`
    let y: &mut *mut () = &mut std::ptr::null_mut();
    //~^ NOTE has type `&mut *mut ()` which is not `Send`
    assert_send(
        //~^ ERROR coroutine cannot be sent between threads safely
        //~| NOTE coroutine is not `Send`
        #[coroutine]
        move || {
            //~^ ERROR coroutine cannot be sent between threads safely
            //~| NOTE coroutine is not `Send`
            //~| NOTE coroutine is not `Send`
            //~| yield occurs here
            yield;
            let _x = x;
            //~^ NOTE captured value is not `Send`
            //~| NOTE has type `&*mut ()` which is not `Send`, because `*mut ()` is not `Sync`
        },
    );
    assert_send(
        //~^ ERROR coroutine cannot be sent between threads safely
        //~| NOTE coroutine is not `Send`
        #[coroutine]
        move || {
            //~^ ERROR coroutine cannot be sent between threads safely
            //~| NOTE coroutine is not `Send`
            //~| NOTE coroutine is not `Send`
            //~| yield occurs here
            yield;
            let _y = y;
            //~^ NOTE captured value is not `Send`
            //~| NOTE has type `&mut *mut ()` which is not `Send`, because `*mut ()` is not `Send`
        },
    );
}
