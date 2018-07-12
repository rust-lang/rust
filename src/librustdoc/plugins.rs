// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(deprecated)]

use clean;

pub type PluginResult = clean::Crate;
pub type PluginCallback = fn (clean::Crate) -> PluginResult;

/// Manages loading and running of plugins
pub struct PluginManager {
    callbacks: Vec<PluginCallback> ,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> PluginManager {
        PluginManager {
            callbacks: Vec::new(),
        }
    }

    /// Load a normal Rust function as a plugin.
    ///
    /// This is to run passes over the cleaned crate. Plugins run this way
    /// correspond to the A-aux tag on Github.
    pub fn add_plugin(&mut self, plugin: PluginCallback) {
        self.callbacks.push(plugin);
    }
    /// Run all the loaded plugins over the crate, returning their results
    pub fn run_plugins(&self, mut krate: clean::Crate) -> clean::Crate {
        for &callback in &self.callbacks {
            krate = callback(krate);
        }
        krate
    }
}
