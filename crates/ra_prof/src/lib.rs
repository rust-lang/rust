use std::cell::RefCell;
use std::time::{Duration, Instant};
use std::mem;
use std::io::{stderr, Write};
use std::iter::repeat;
use std::collections::{HashSet};
use std::default::Default;
use std::iter::FromIterator;
use std::sync::RwLock;
use lazy_static::lazy_static;

/// Set profiling filter. It specifies descriptions allowed to profile.
/// This is helpful when call stack has too many nested profiling scopes.
/// Additionally filter can specify maximum depth of profiling scopes nesting.
///
/// #Example
/// ```
/// use ra_prof::set_filter;
/// use ra_prof::Filter;
/// let max_depth = 2;
/// let allowed = vec!["profile1".to_string(), "profile2".to_string()];
/// let f = Filter::new( max_depth, allowed );
/// set_filter(f);
/// ```
///
pub fn set_filter(f: Filter) {
    let set = HashSet::from_iter(f.allowed.iter().cloned());
    let mut old = FILTER.write().unwrap();
    let filter_data = FilterData { depth: f.depth, allowed: set, version: old.version + 1 };
    *old = filter_data;
}

/// This function starts a profiling scope in the current execution stack with a given description.
/// It returns a Profile structure and measure elapsed time between this method invocation and Profile structure drop.
/// It supports nested profiling scopes in case when this function invoked multiple times at the execution stack. In this case the profiling information will be nested at the output.
/// Profiling information is being printed in the stderr.
///
/// #Example
/// ```
/// use ra_prof::profile;
/// use ra_prof::set_filter;
/// use ra_prof::Filter;
///
/// let allowed = vec!["profile1".to_string(), "profile2".to_string()];
/// let f = Filter::new(2, allowed);
/// set_filter(f);
/// profiling_function1();
///
/// fn profiling_function1() {
///     let _p = profile("profile1");
///     profiling_function2();
/// }
///
/// fn profiling_function2() {
///     let _p = profile("profile2");
/// }
/// ```
/// This will print in the stderr the following:
/// ```text
///  0ms - profile
///      0ms - profile2
/// ```
///
pub fn profile(desc: &str) -> Profiler {
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
            Profiler { desc: None }
        } else if stack.starts.len() < stack.filter_data.depth
            && stack.filter_data.allowed.contains(&desc_str)
        {
            stack.starts.push(Instant::now());
            Profiler { desc: Some(desc_str) }
        } else {
            Profiler { desc: None }
        }
    })
}

pub struct Profiler {
    desc: Option<String>,
}

pub struct Filter {
    depth: usize,
    allowed: Vec<String>,
}

impl Filter {
    pub fn new(depth: usize, allowed: Vec<String>) -> Filter {
        Filter { depth, allowed }
    }
}

struct ProfileStack {
    starts: Vec<Instant>,
    messages: Vec<Message>,
    filter_data: FilterData,
}

struct Message {
    level: usize,
    duration: Duration,
    message: String,
}

impl ProfileStack {
    fn new() -> ProfileStack {
        ProfileStack { starts: Vec::new(), messages: Vec::new(), filter_data: Default::default() }
    }
}

#[derive(Default, Clone)]
struct FilterData {
    depth: usize,
    version: usize,
    allowed: HashSet<String>,
}

lazy_static! {
    static ref FILTER: RwLock<FilterData> = RwLock::new(Default::default());
}

thread_local!(static PROFILE_STACK: RefCell<ProfileStack> = RefCell::new(ProfileStack::new()));

impl Drop for Profiler {
    fn drop(&mut self) {
        match self {
            Profiler { desc: Some(desc) } => {
                PROFILE_STACK.with(|stack| {
                    let mut stack = stack.borrow_mut();
                    let start = stack.starts.pop().unwrap();
                    let duration = start.elapsed();
                    let level = stack.starts.len();
                    let message = mem::replace(desc, String::new());
                    stack.messages.push(Message { level, duration, message });
                    if level == 0 {
                        let stdout = stderr();
                        print(0, &stack.messages, &mut stdout.lock());
                        stack.messages.clear();
                    }
                });
            }
            Profiler { desc: None } => (),
        }
    }
}

fn print(lvl: usize, msgs: &[Message], out: &mut impl Write) {
    let mut last = 0;
    let indent = repeat("    ").take(lvl + 1).collect::<String>();
    for (i, &Message { level: l, duration: dur, message: ref msg }) in msgs.iter().enumerate() {
        if l != lvl {
            continue;
        }
        writeln!(out, "{} {:6}ms - {}", indent, dur.as_millis(), msg)
            .expect("printing profiling info to stdout");

        print(lvl + 1, &msgs[last..i], out);
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
        let f = Filter::new(2, s);
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
