// edition:2021

#![feature(inline_const)]

fn closure() {
    loop {
        let closure = || {
            if true {
                Err(1)
                //~^ ERROR mismatched types
            }

            Ok(())
        };
    }
}

fn async_block() {
    loop {
        let fut = async {
            if true {
                Err(1)
                //~^ ERROR mismatched types
            }

            Ok(())
        };
    }
}

fn fn_item() {
    let _ = loop {
        fn foo() -> Result<(), ()> {
            if true {
                Err(1)
                //~^ ERROR mismatched types
            }
            Err(())
        }
    };
}

fn const_block() {
    let _ = loop {
        const {
            if true {
                Err(1)
                //~^ ERROR mismatched types
            }
            Err(())
        };
    };
}

fn main() {}
