// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Simple metrics gathering and reporting
 *
 * This is inspired from https://github.com/rcrowley/go-metrics and
 * https://github.com/codahale/metrics.
 */

use std::local_data;
use std::hashmap::HashMap;
use std::container::MutableMap;

pub static reg_key: local_data::Key<HashMap<~str, MetricGroup>> = &local_data::Key;

/// A group of metrics with a common namespace
pub struct MetricGroup {
    priv name: ~str,
    priv metrics: HashMap<~str, Metric>,
}

impl MetricGroup {
    /// Create a new MetricGroup with the given namespace
    pub fn new(name: ~str) -> MetricGroup {
        MetricGroup { name: name, metrics: HashMap::new() }
    }

    /// Add a metric as a child with the given name.
    pub fn insert(&mut self, name: ~str, m: Metric) {
        self.metrics.insert(name, m);
    }
}

// TLS interface

/// Adds a metric to TLS
///
/// The name consists of any string, a period, and any string that doesn't
/// contain a period. Example: `rustc.trans.insns.pop`. `rustc.trans.insns` is
/// the namespace, `pop` is the name of the metric.
pub fn add(name: ~str, m: Metric) {
    let last = name.rfind('.').expect("all metrics require a namespace");
    let prefix = name.slice_to(last);
    let key = name.slice_from(last);

    do local_data::modify(reg_key) |reg| {
        let mut reg = reg.unwrap_or_default(HashMap::new());
        if reg.contains_key_equiv(&prefix) {
            // ugly allocation
            let mg = reg.get_mut(&prefix.to_str());
            mg.metrics.insert(key.to_str(), m);
        } else {
            let mut mg = MetricGroup::new(prefix.to_str());
            mg.metrics.insert(key.to_str(), m);
            reg.insert(prefix.to_str(), mg);
        }
        Some(reg)
    }
}

/// Modify a metric in TLS
///
/// The name is the name of a metric that may or may not have been added
/// already. The passed closure receives a mutable reference to the metric, or
/// None.
pub fn modify(name: ~str, f: &fn(m: Option<&mut Metric>)) {
    let last = name.rfind('.').expect("all metrics require a namespace");
    let prefix = name.slice_to(last);
    let key = name.slice_from(last);
    
    do local_data::get_mut(reg_key) |reg| {
        match reg {
            Some(reg) => {
                if reg.contains_key_equiv(&prefix) {
                    let mg = reg.get_mut(&prefix.to_str());
                    f(mg.metrics.find_mut(&key.to_str()));
                } else {
                    f(None);
                }
            },
            None => f(None)
        }
    }
}

/// Get a metric from TLS
///
/// The name is the name of a metric that may or may not have been added
/// already. The passed closure receives a reference to the metric, or None.
pub fn get(name: ~str, f: &fn(m: Option<&Metric>)) {
    let last = name.rfind('.').expect("all metrics require a namespace");
    let prefix = name.slice_to(last);
    let key = name.slice_from(last);
    
    do local_data::get(reg_key) |reg| {
        match reg {
            Some(reg) => {
                if reg.contains_key_equiv(&prefix) {
                    let mg = reg.get(&prefix.to_str());
                    f(mg.metrics.find(&key.to_str()));
                } else {
                    f(None);
                }
            },
            None => f(None)
        }
    }
}
/// The set of metrics which this library knows about
pub enum Metric {
    Counter(Counter),
}

/// A counter which increments and decrements over time
pub struct Counter {
    priv val: i64
}

// xxx: wrapping
impl Counter {
    /// Create a new counter
    pub fn new   ()      -> Counter      { Counter { val: 0 } }
    /// Get the value of the counter
    pub fn get   (&self) -> i64          { self.val }
    /// Set the counter to 0
    pub fn reset (&mut self)             { self.val = 0; }
    /// Increment the counter by `value`
    pub fn inc   (&mut self, value: i64) { self.val += value; }
    /// Decrement the counter by `value`
    pub fn dec   (&mut self, value: i64) { self.val -= value; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let mut c = Counter::new();
        assert_eq!(c.get(), 0);
        c.inc(5);
        assert_eq!(c.get(), 5);
        c.dec(2);
        assert_eq!(c.get(), 3);
        c.reset();
        assert_eq!(c.get(), 0);
    }

    #[test]
    fn test_tls() {
        let m = Counter(Counter::new());
        add(~"test.counter", m);
        modify(~"test.counter", |m| m.unwrap().inc(1));
        get(~"test.counter", |m| assert_eq!(match m.unwrap() { &Counter(ref c) => c.get() }, 1));
    }
}
