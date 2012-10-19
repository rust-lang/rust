// helper for transmutation, shown below.
type RustClosure = (int,int);

struct Condition<T, U:Copy> {
    key: task::local_data::LocalDataKey<Handler<T,U>>
}

struct Handler<T, U:Copy> {
    handle: RustClosure
}


struct ProtectBlock<T, U:Copy> {
    cond: &Condition<T, U>,
    inner: RustClosure
}

struct Guard<T, U:Copy> {
    cond: &Condition<T,U>,
    prev: Option<@Handler<T, U>>,
    drop {
        match self.prev {
            None => (),
            Some(p) =>
            unsafe {
                debug!("Guard: popping handler from TLS");
                task::local_data::local_data_set(self.cond.key, p)
            }
        }
    }
}

struct HandleBlock<T, U:Copy> {
    pb: &ProtectBlock<T,U>,
    prev: Option<@Handler<T,U>>,
    handler: @Handler<T,U>,
    drop {
        unsafe {
            debug!("HandleBlock: pushing handler to TLS");
            let _g = Guard { cond: self.pb.cond,
                             prev: self.prev };
            task::local_data::local_data_set(self.pb.cond.key,
                                             self.handler);
            // transmutation to avoid copying non-copyable, should
            // be fixable by tracking closure pointees in regionck.
            let f : &fn() = ::cast::transmute(self.pb.inner);
            debug!("HandleBlock: invoking protected code");
            f();
            debug!("HandleBlock: returned from protected code");
        }
    }
}

impl<T, U: Copy> ProtectBlock<T,U> {
    fn handle(&self, h: &self/fn(&T) ->U) -> HandleBlock/&self<T,U> {
        unsafe {
            debug!("ProtectBlock.handle: setting up handler block");
            let p : *RustClosure = ::cast::transmute(&h);
            let prev = task::local_data::local_data_get(self.cond.key);
            HandleBlock { pb: self,
                          prev: prev,
                          handler: @Handler{handle: *p} }
        }
    }
}


impl<T, U: Copy>  Condition<T,U> {

    fn guard(&self, h: &self/fn(&T) ->U) -> Guard/&self<T,U> {
        unsafe {
            let prev = task::local_data::local_data_get(self.key);
            let g = Guard { cond: self, prev: prev };
            debug!("Guard: pushing handler to TLS");
            let p : *RustClosure = ::cast::transmute(&h);
            let h = @Handler{handle: *p};
            task::local_data::local_data_set(self.key, h);
            move g
        }
    }

    fn protect(&self, inner: &self/fn()) -> ProtectBlock/&self<T,U> {
        unsafe {
            // transmutation to avoid copying non-copyable, should
            // be fixable by tracking closure pointees in regionck.
            debug!("Condition.protect: setting up protected block");
            let p : *RustClosure = ::cast::transmute(&inner);
            ProtectBlock { cond: self,
                           inner: *p }
        }
    }

    fn raise(t:&T) -> U {
        unsafe {
            match task::local_data::local_data_get(self.key) {
                None => {
                    debug!("Condition.raise: found no handler");
                    fail
                }

                Some(handler) => {
                    debug!("Condition.raise: found handler");
                    let f : &fn(&T) -> U = ::cast::transmute(handler.handle);
                    f(t)
                }
            }
        }
    }
}


#[cfg(test)]
fn sadness_key(_x: @Handler<int,int>) { }

#[cfg(test)]
fn trouble(i: int) {
    // Condition should work as a const, just limitations in consts.
    let sadness_condition : Condition<int,int> =
        Condition { key: sadness_key };
    debug!("trouble: raising conition");
    let j = sadness_condition.raise(&i);
    debug!("trouble: handler recovered with %d", j);
}

#[test]
fn test1() {

    let sadness_condition : Condition<int,int> =
        Condition { key: sadness_key };

    let mut i = 10;

    let b = do sadness_condition.protect {
        debug!("test1: in protected block");
        trouble(1);
        trouble(2);
        trouble(3);
    };

    do b.handle |j| {
        debug!("test1: in handler");
        i += *j;
        i
    };

    assert i == 16;
}
#[cfg(test)]
fn nested_test_inner() {
    let sadness_condition : Condition<int,int> =
        Condition { key: sadness_key };

    let mut inner_trapped = false;

    let b = do sadness_condition.protect {
        debug!("nested_test_inner: in protected block");
        trouble(1);
    };

    do b.handle |_j| {
        debug!("nested_test_inner: in handler");
        inner_trapped = true;
        0
    };

    assert inner_trapped;
}

#[test]
fn nested_test_outer() {

    let sadness_condition : Condition<int,int> =
        Condition { key: sadness_key };

    let mut outer_trapped = false;

    let b = do sadness_condition.protect {
        debug!("nested_test_outer: in protected block");
        nested_test_inner();
        trouble(1);
    };

    do b.handle |_j| {
        debug!("nested_test_outer: in handler");
        outer_trapped = true;
        0
    };

    assert outer_trapped;
}


#[cfg(test)]
fn nested_guard_test_inner() {
    let sadness_condition : Condition<int,int> =
        Condition { key: sadness_key };

    let mut inner_trapped = false;

    let _g = do sadness_condition.guard |_j| {
        debug!("nested_guard_test_inner: in handler");
        inner_trapped = true;
        0
    };

    debug!("nested_guard_test_inner: in protected block");
    trouble(1);

    assert inner_trapped;
}

#[test]
fn nested_guard_test_outer() {

    let sadness_condition : Condition<int,int> =
        Condition { key: sadness_key };

    let mut outer_trapped = false;

    let _g = do sadness_condition.guard |_j| {
        debug!("nested_guard_test_outer: in handler");
        outer_trapped = true;
        0
    };

    debug!("nested_guard_test_outer: in protected block");
    nested_guard_test_inner();
    trouble(1);

    assert outer_trapped;
}
