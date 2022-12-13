// edition:2018

fn is_unwindsafe(_: impl std::panic::UnwindSafe) {}

fn main() {
    // a normal async block that uses `&mut Context<'_>` implicitly via async lowering is fine
    // as we should not consider that to be alive across an await point
    is_unwindsafe(async {
        async {}.await; // this needs an inner await point
    });

    is_unwindsafe(async {
        //~^ ERROR the type `&mut Context<'_>` may not be safely transferred across an unwind boundary
        use std::ptr::null;
        use std::task::{Context, RawWaker, RawWakerVTable, Waker};
        let waker = unsafe {
            Waker::from_raw(RawWaker::new(
                null(),
                &RawWakerVTable::new(|_| todo!(), |_| todo!(), |_| todo!(), |_| todo!()),
            ))
        };
        let mut cx = Context::from_waker(&waker);
        let cx_ref = &mut cx;

        async {}.await; // this needs an inner await point

        // in this case, `&mut Context<'_>` is *truely* alive across an await point
        drop(cx_ref);
        drop(cx);
    });
}
