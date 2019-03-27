extern crate lazy_static;

use std::cell::RefCell;
use std::time;
use std::fmt;
use std::mem;
use std::io::{stderr, StderrLock, Write};
use std::iter::repeat;
use std::collections::{HashSet};
use std::default::Default;
use std::iter::FromIterator;
use std::sync::RwLock;

use lazy_static::lazy_static;

type Message = (usize, u64, String);

pub struct Profiler {
    desc: String,
}

pub struct Filter {
    depth: usize,
    allowed: Vec<String>,
}

struct ProfileStack {
    starts: Vec<time::Instant>,
    messages: Vec<Message>,
    filter_data: FilterData,
}

impl ProfileStack {
    fn new() -> ProfileStack {
        ProfileStack { starts: Vec::new(), messages: Vec::new(), filter_data: Default::default() }
    }
}

#[derive(Default)]
struct FilterData {
    depth: usize,
    version: usize,
    allowed: HashSet<String>,
}

impl Clone for FilterData {
    fn clone(&self) -> FilterData {
        let set = HashSet::from_iter(self.allowed.iter().cloned());
        FilterData { depth: self.depth, allowed: set, version: self.version }
    }
}

lazy_static! {
    static ref FILTER: RwLock<FilterData> = RwLock::new(Default::default());
}

thread_local!(static PROFILE_STACK: RefCell<ProfileStack> = RefCell::new(ProfileStack::new()));

pub fn set_filter(f: Filter) {
    let mut old = FILTER.write().unwrap();
    let set = HashSet::from_iter(f.allowed.iter().cloned());
    let filter_data = FilterData { depth: f.depth, allowed: set, version: old.version + 1 };
    *old = filter_data;
}

pub fn profile<T: fmt::Display>(desc: T) -> Profiler {
    PROFILE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if stack.starts.len() == 0 {
            match FILTER.try_read() {
                Ok(f) => {
                    if f.version > stack.filter_data.version {
                        stack.filter_data = f.clone();
                    }
                }
                Err(_) => (),
            };
        }
        let desc_str = desc.to_string();
        if desc_str.is_empty() {
            Profiler { desc: desc_str }
        } else if stack.starts.len() < stack.filter_data.depth
            && stack.filter_data.allowed.contains(&desc_str)
        {
            stack.starts.push(time::Instant::now());
            Profiler { desc: desc_str }
        } else {
            Profiler { desc: String::new() }
        }
    })
}

impl Drop for Profiler {
    fn drop(&mut self) {
        if self.desc.is_empty() {
            return;
        }
        PROFILE_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            let start = stack.starts.pop().unwrap();
            let duration = start.elapsed();
            let duration_ms = duration.as_secs() * 1000 + u64::from(duration.subsec_millis());
            let stack_len = stack.starts.len();
            let msg = (stack_len, duration_ms, mem::replace(&mut self.desc, String::new()));
            stack.messages.push(msg);
            if stack_len == 0 {
                let stdout = stderr();
                print(0, &stack.messages, 1, &mut stdout.lock());
                stack.messages.clear();
            }
        });
    }
}

fn print(lvl: usize, msgs: &[Message], enabled: usize, stdout: &mut StderrLock<'_>) {
    if lvl > enabled {
        return;
    }
    let mut last = 0;
    for (i, &(l, time, ref msg)) in msgs.iter().enumerate() {
        if l != lvl {
            continue;
        }
        writeln!(
            stdout,
            "{} {:6}ms - {}",
            repeat("    ").take(lvl + 1).collect::<String>(),
            time,
            msg
        )
        .expect("printing profiling info to stdout");

        print(lvl + 1, &msgs[last..i], enabled, stdout);
        last = i;
    }
}

#[cfg(test)]
mod tests {

    use super::profile;
    use super::set_filter;
    use super::Filter;

    #[test]
    fn test_basic_profile() {
        let s = vec!["profile1".to_string(), "profile2".to_string()];
        let f = Filter { depth: 2, allowed: s };
        set_filter(f);
        profiling_function1();
    }

    fn profiling_function1() {
        let _p = profile("profile1");
        profiling_function2();
    }

    fn profiling_function2() {
        let _p = profile("profile2");
    }
}
