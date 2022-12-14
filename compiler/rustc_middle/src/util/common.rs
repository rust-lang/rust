use rustc_data_structures::sync::Lock;

use std::fmt::Debug;
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests;

pub fn to_readable_str(mut val: usize) -> String {
    let mut groups = vec![];
    loop {
        let group = val % 1000;

        val /= 1000;

        if val == 0 {
            groups.push(group.to_string());
            break;
        } else {
            groups.push(format!("{:03}", group));
        }
    }

    groups.reverse();

    groups.join("_")
}

pub fn record_time<T, F>(accu: &Lock<Duration>, f: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let rv = f();
    let duration = start.elapsed();
    let mut accu = accu.lock();
    *accu += duration;
    rv
}

pub fn indent<R, F>(op: F) -> R
where
    R: Debug,
    F: FnOnce() -> R,
{
    // Use in conjunction with the log post-processor like `src/etc/indenter`
    // to make debug output more readable.
    debug!(">>");
    let r = op();
    debug!("<< (Result = {:?})", r);
    r
}

pub struct Indenter {
    _cannot_construct_outside_of_this_module: (),
}

impl Drop for Indenter {
    fn drop(&mut self) {
        debug!("<<");
    }
}

pub fn indenter() -> Indenter {
    debug!(">>");
    Indenter { _cannot_construct_outside_of_this_module: () }
}
