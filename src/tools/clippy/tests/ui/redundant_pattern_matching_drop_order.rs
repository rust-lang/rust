// Issue #5746
#![warn(clippy::redundant_pattern_matching)]
#![allow(
    clippy::if_same_then_else,
    clippy::equatable_if_let,
    clippy::needless_if,
    clippy::needless_else
)]
use std::task::Poll::{Pending, Ready};

fn main() {
    let m = std::sync::Mutex::new((0, 0));

    // Result
    if let Ok(_) = m.lock() {}
    //~^ redundant_pattern_matching
    if let Err(_) = Err::<(), _>(m.lock().unwrap().0) {}
    //~^ redundant_pattern_matching

    {
        if let Ok(_) = Ok::<_, std::sync::MutexGuard<()>>(()) {}
        //~^ redundant_pattern_matching
    }
    if let Ok(_) = Ok::<_, std::sync::MutexGuard<()>>(()) {
        //~^ redundant_pattern_matching
    } else {
    }
    if let Ok(_) = Ok::<_, std::sync::MutexGuard<()>>(()) {}
    //~^ redundant_pattern_matching
    if let Err(_) = Err::<std::sync::MutexGuard<()>, _>(()) {}
    //~^ redundant_pattern_matching

    if let Ok(_) = Ok::<_, ()>(String::new()) {}
    //~^ redundant_pattern_matching
    if let Err(_) = Err::<(), _>((String::new(), ())) {}
    //~^ redundant_pattern_matching

    // Option
    if let Some(_) = Some(m.lock()) {}
    //~^ redundant_pattern_matching
    if let Some(_) = Some(m.lock().unwrap().0) {}
    //~^ redundant_pattern_matching

    {
        if let None = None::<std::sync::MutexGuard<()>> {}
        //~^ redundant_pattern_matching
    }
    if let None = None::<std::sync::MutexGuard<()>> {
        //~^ redundant_pattern_matching
    } else {
    }

    if let None = None::<std::sync::MutexGuard<()>> {}
    //~^ redundant_pattern_matching

    if let Some(_) = Some(String::new()) {}
    //~^ redundant_pattern_matching
    if let Some(_) = Some((String::new(), ())) {}
    //~^ redundant_pattern_matching

    // Poll
    if let Ready(_) = Ready(m.lock()) {}
    //~^ redundant_pattern_matching
    if let Ready(_) = Ready(m.lock().unwrap().0) {}
    //~^ redundant_pattern_matching

    {
        if let Pending = Pending::<std::sync::MutexGuard<()>> {}
        //~^ redundant_pattern_matching
    }
    if let Pending = Pending::<std::sync::MutexGuard<()>> {
        //~^ redundant_pattern_matching
    } else {
    }

    if let Pending = Pending::<std::sync::MutexGuard<()>> {}
    //~^ redundant_pattern_matching

    if let Ready(_) = Ready(String::new()) {}
    //~^ redundant_pattern_matching
    if let Ready(_) = Ready((String::new(), ())) {}
    //~^ redundant_pattern_matching
}
