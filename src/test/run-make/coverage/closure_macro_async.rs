// compile-flags: --edition=2018
#![feature(no_coverage)]

macro_rules! bail {
    ($msg:literal $(,)?) => {
        if $msg.len() > 0 {
            println!("no msg");
        } else {
            println!($msg);
        }
        return Err(String::from($msg));
    };
}

macro_rules! on_error {
    ($value:expr, $error_message:expr) => {
        $value.or_else(|e| { // FIXME(85000): no coverage in closure macros
            let message = format!($error_message, e);
            if message.len() > 0 {
                println!("{}", message);
                Ok(String::from("ok"))
            } else {
                bail!("error");
            }
        })
    };
}

fn load_configuration_files() -> Result<String, String> {
    Ok(String::from("config"))
}

pub async fn test() -> Result<(), String> {
    println!("Starting service");
    let config = on_error!(load_configuration_files(), "Error loading configs: {}")?;

    let startup_delay_duration = String::from("arg");
    let _ = (config, startup_delay_duration);
    Ok(())
}

#[no_coverage]
fn main() {
    executor::block_on(test());
}

mod executor {
    use core::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    };

    #[no_coverage]
    pub fn block_on<F: Future>(mut future: F) -> F::Output {
        let mut future = unsafe { Pin::new_unchecked(&mut future) };
        use std::hint::unreachable_unchecked;
        static VTABLE: RawWakerVTable = RawWakerVTable::new(

            #[no_coverage]
            |_| unsafe { unreachable_unchecked() }, // clone

            #[no_coverage]
            |_| unsafe { unreachable_unchecked() }, // wake

            #[no_coverage]
            |_| unsafe { unreachable_unchecked() }, // wake_by_ref

            #[no_coverage]
            |_| (),
        );
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) };
        let mut context = Context::from_waker(&waker);

        loop {
            if let Poll::Ready(val) = future.as_mut().poll(&mut context) {
                break val;
            }
        }
    }
}
