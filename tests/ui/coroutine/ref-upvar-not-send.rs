// For `Send` coroutines, suggest a `T: Sync` requirement for `&T` upvars,
// and suggest a `T: Send` requirement for `&mut T` upvars.

#![feature(coroutines, stmt_expr_attributes)]

fn assert_send<T: Send>(_: T) {}
//~^ NOTE required by a bound in `assert_send`
//~| NOTE required by this bound in `assert_send`
//~| NOTE required by a bound in `assert_send`
//~| NOTE required by this bound in `assert_send`

fn main() {
    let x: &*mut () = &std::ptr::null_mut();
    //~^ NOTE has type `&*mut ()` which is not `Send`
    //~| NOTE captured value is not `Send` because `&` references cannot be sent unless their referent is `Sync`
    let y: &mut *mut () = &mut std::ptr::null_mut();
    //~^ NOTE has type `&mut *mut ()` which is not `Send`
    //~| NOTE captured value is not `Send` because `&mut` references cannot be sent unless their referent is `Send`
    assert_send(
        //~^ ERROR coroutine cannot be sent between threads safely
        //~| NOTE coroutine is not `Send`
        #[coroutine]
        move || {
            yield;
            let _x = x;
        },
    );
    assert_send(
        //~^ ERROR coroutine cannot be sent between threads safely
        //~| NOTE coroutine is not `Send`
        #[coroutine]
        move || {
            yield;
            let _y = y;
        },
    );
}
