//! Regression test for https://github.com/rust-lang/rust/issues/127048.
//! A nested initializer block should still attribute a mismatch to the local type.

//@ edition: 2021

fn direct_initializer() {
    || {
        Ok::<(), ()>(())?;
        let _: i32 = false;
        //~^ ERROR mismatched types
        todo!()
    };
}

fn nested_initializer() {
    || {
        Ok::<(), ()>(())?;
        let _: i32 = {
            false
            //~^ ERROR mismatched types
        };
        todo!()
    };
}

fn explicit_closure_return_type() {
    || -> Result<(), ()> {
        Ok::<(), ()>(())?;
        let _: i32 = {
            false
            //~^ ERROR mismatched types
        };
        todo!()
    };
}

fn async_block() {
    async {
        Ok::<(), ()>(())?;
        let _: i32 = {
            false
            //~^ ERROR mismatched types
        };
        todo!()
    };
}

fn assignment_rhs() {
    let mut value: i32 = 0;
    value = false;
    //~^ ERROR mismatched types
    value = {
        let _ = ();
        false
        //~^ ERROR mismatched types
    };
}

fn binary_rhs() {
    let _ = true && 1;
    //~^ ERROR mismatched types
    let _ = true && {
        let _ = ();
        1
        //~^ ERROR mismatched types
    };
}

fn main() {}
