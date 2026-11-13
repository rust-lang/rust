// Repro for <https://github.com/rust-lang/rust/issues/71671#issuecomment-848994782>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

pub trait Robot {
    type Id;
}

pub type DynRobot = Box<dyn Robot<Id = u32> + Send>;

impl Robot for DynRobot {
    type Id = u32;
}

struct IRobot<R: Robot> {
    id: R::Id,
    robot: R,
}

// stand-in for tokio::spawn
fn this_is_send<T: Send>(value: T) -> T {
    value
}

async fn yield_now() {}

fn test(source: DynRobot) {
    let _my_task = this_is_send(async move {
        let _my_iter = IRobot {
            id: 32,
            robot: source,
        };
        yield_now().await;
    });
}

fn main() {}
