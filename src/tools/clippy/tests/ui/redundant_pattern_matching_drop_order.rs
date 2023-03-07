// run-rustfix

// Issue #5746
#![warn(clippy::redundant_pattern_matching)]
#![allow(clippy::if_same_then_else, clippy::equatable_if_let)]
use std::task::Poll::{Pending, Ready};

fn main() {
    let m = std::sync::Mutex::new((0, 0));

    // Result
    if let Ok(_) = m.lock() {}
    if let Err(_) = Err::<(), _>(m.lock().unwrap().0) {}

    {
        if let Ok(_) = Ok::<_, std::sync::MutexGuard<()>>(()) {}
    }
    if let Ok(_) = Ok::<_, std::sync::MutexGuard<()>>(()) {
    } else {
    }
    if let Ok(_) = Ok::<_, std::sync::MutexGuard<()>>(()) {}
    if let Err(_) = Err::<std::sync::MutexGuard<()>, _>(()) {}

    if let Ok(_) = Ok::<_, ()>(String::new()) {}
    if let Err(_) = Err::<(), _>((String::new(), ())) {}

    // Option
    if let Some(_) = Some(m.lock()) {}
    if let Some(_) = Some(m.lock().unwrap().0) {}

    {
        if let None = None::<std::sync::MutexGuard<()>> {}
    }
    if let None = None::<std::sync::MutexGuard<()>> {
    } else {
    }

    if let None = None::<std::sync::MutexGuard<()>> {}

    if let Some(_) = Some(String::new()) {}
    if let Some(_) = Some((String::new(), ())) {}

    // Poll
    if let Ready(_) = Ready(m.lock()) {}
    if let Ready(_) = Ready(m.lock().unwrap().0) {}

    {
        if let Pending = Pending::<std::sync::MutexGuard<()>> {}
    }
    if let Pending = Pending::<std::sync::MutexGuard<()>> {
    } else {
    }

    if let Pending = Pending::<std::sync::MutexGuard<()>> {}

    if let Ready(_) = Ready(String::new()) {}
    if let Ready(_) = Ready((String::new(), ())) {}
}
