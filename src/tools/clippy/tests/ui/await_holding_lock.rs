#![warn(clippy::await_holding_lock)]
#![allow(clippy::readonly_write_lock)]

// When adding or modifying a test, please do the same for parking_lot::Mutex.
mod std_mutex {
    use super::baz;
    use std::sync::{Mutex, RwLock};

    pub async fn bad(x: &Mutex<u32>) -> u32 {
        let guard = x.lock().unwrap();
        //~^ ERROR: this `MutexGuard` is held across an await point
        baz().await
    }

    pub async fn good(x: &Mutex<u32>) -> u32 {
        {
            let guard = x.lock().unwrap();
            let y = *guard + 1;
        }
        baz().await;
        let guard = x.lock().unwrap();
        47
    }

    pub async fn bad_rw(x: &RwLock<u32>) -> u32 {
        let guard = x.read().unwrap();
        //~^ ERROR: this `MutexGuard` is held across an await point
        baz().await
    }

    pub async fn bad_rw_write(x: &RwLock<u32>) -> u32 {
        let mut guard = x.write().unwrap();
        //~^ ERROR: this `MutexGuard` is held across an await point
        baz().await
    }

    pub async fn good_rw(x: &RwLock<u32>) -> u32 {
        {
            let guard = x.read().unwrap();
            let y = *guard + 1;
        }
        {
            let mut guard = x.write().unwrap();
            *guard += 1;
        }
        baz().await;
        let guard = x.read().unwrap();
        47
    }

    pub async fn also_bad(x: &Mutex<u32>) -> u32 {
        let first = baz().await;

        let guard = x.lock().unwrap();
        //~^ ERROR: this `MutexGuard` is held across an await point

        let second = baz().await;

        let third = baz().await;

        first + second + third
    }

    pub async fn not_good(x: &Mutex<u32>) -> u32 {
        let first = baz().await;

        let second = {
            let guard = x.lock().unwrap();
            //~^ ERROR: this `MutexGuard` is held across an await point
            baz().await
        };

        let third = baz().await;

        first + second + third
    }

    #[allow(clippy::manual_async_fn)]
    pub fn block_bad(x: &Mutex<u32>) -> impl std::future::Future<Output = u32> + '_ {
        async move {
            let guard = x.lock().unwrap();
            //~^ ERROR: this `MutexGuard` is held across an await point
            baz().await
        }
    }
}

// When adding or modifying a test, please do the same for std::Mutex.
mod parking_lot_mutex {
    use super::baz;
    use parking_lot::{Mutex, RwLock};

    pub async fn bad(x: &Mutex<u32>) -> u32 {
        let guard = x.lock();
        //~^ ERROR: this `MutexGuard` is held across an await point
        baz().await
    }

    pub async fn good(x: &Mutex<u32>) -> u32 {
        {
            let guard = x.lock();
            let y = *guard + 1;
        }
        baz().await;
        let guard = x.lock();
        47
    }

    pub async fn bad_rw(x: &RwLock<u32>) -> u32 {
        let guard = x.read();
        //~^ ERROR: this `MutexGuard` is held across an await point
        baz().await
    }

    pub async fn bad_rw_write(x: &RwLock<u32>) -> u32 {
        let mut guard = x.write();
        //~^ ERROR: this `MutexGuard` is held across an await point
        baz().await
    }

    pub async fn good_rw(x: &RwLock<u32>) -> u32 {
        {
            let guard = x.read();
            let y = *guard + 1;
        }
        {
            let mut guard = x.write();
            *guard += 1;
        }
        baz().await;
        let guard = x.read();
        47
    }

    pub async fn also_bad(x: &Mutex<u32>) -> u32 {
        let first = baz().await;

        let guard = x.lock();
        //~^ ERROR: this `MutexGuard` is held across an await point

        let second = baz().await;

        let third = baz().await;

        first + second + third
    }

    pub async fn not_good(x: &Mutex<u32>) -> u32 {
        let first = baz().await;

        let second = {
            let guard = x.lock();
            //~^ ERROR: this `MutexGuard` is held across an await point
            baz().await
        };

        let third = baz().await;

        first + second + third
    }

    #[allow(clippy::manual_async_fn)]
    pub fn block_bad(x: &Mutex<u32>) -> impl std::future::Future<Output = u32> + '_ {
        async move {
            let guard = x.lock();
            //~^ ERROR: this `MutexGuard` is held across an await point
            baz().await
        }
    }
}

async fn baz() -> u32 {
    42
}

async fn no_await(x: std::sync::Mutex<u32>) {
    let mut guard = x.lock().unwrap();
    *guard += 1;
}

// FIXME: FP, because the `MutexGuard` is dropped before crossing the await point. This is
// something the needs to be fixed in rustc. There's already drop-tracking, but this is currently
// disabled, see rust-lang/rust#93751. This case isn't picked up by drop-tracking though. If the
// `*guard += 1` is removed it is picked up.
async fn dropped_before_await(x: std::sync::Mutex<u32>) {
    let mut guard = x.lock().unwrap();
    //~^ ERROR: this `MutexGuard` is held across an await point
    *guard += 1;
    drop(guard);
    baz().await;
}

fn main() {
    let m = std::sync::Mutex::new(100);
    std_mutex::good(&m);
    std_mutex::bad(&m);
    std_mutex::also_bad(&m);
    std_mutex::not_good(&m);
    std_mutex::block_bad(&m);

    let m = parking_lot::Mutex::new(100);
    parking_lot_mutex::good(&m);
    parking_lot_mutex::bad(&m);
    parking_lot_mutex::also_bad(&m);
    parking_lot_mutex::not_good(&m);
}
