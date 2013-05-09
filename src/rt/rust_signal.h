// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifndef RUST_SIGNAL_H
#define RUST_SIGNAL_H

// Just an abstract class that represents something that can be signalled
class rust_signal {
public:
    virtual void signal() = 0;
    virtual ~rust_signal() {}
    rust_signal() {}

private:
    // private and undefined to disable copying
    rust_signal(const rust_signal& rhs);
    rust_signal& operator=(const rust_signal& rhs);
};

#endif /* RUST_SIGNAL_H */
