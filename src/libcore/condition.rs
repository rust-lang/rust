// helper for transmutation, shown below.
type RustClosure = (int,int);

struct Condition {
    key: task::local_data::LocalDataKey<Handler>
}

struct Handler {
    handle: RustClosure,
    prev: Option<@Handler>
}


struct ProtectBlock {
    cond: &Condition,
    inner: RustClosure
}

struct PopHandler {
    cond: &Condition,
    drop {
        unsafe {
            debug!("PopHandler: popping handler from TLS");
            match task::local_data::local_data_pop(self.cond.key) {
                None => (),
                Some(h) => {
                    match h.prev {
                        None => (),
                        Some(p) =>
                        task::local_data::local_data_set(self.cond.key, p)
                    }
                }
            }
        }
    }
}

struct HandleBlock {
    pb: &ProtectBlock,
    handler: @Handler,
    drop {
        unsafe {
            debug!("HandleBlock: pushing handler to TLS");
            task::local_data::local_data_set(self.pb.cond.key,
                                             self.handler);
            let _pop = PopHandler { cond: self.pb.cond };
            // transmutation to avoid copying non-copyable, should
            // be fixable by tracking closure pointees in regionck.
            let f : &fn() = ::cast::transmute(self.pb.inner);
            debug!("HandleBlock: invoking protected code");
            f();
            debug!("HandleBlock: returned from protected code");
        }
    }
}

impl ProtectBlock {
    fn handle<T, U: Copy>(&self, h: &self/fn(&T) ->U) -> HandleBlock/&self {
        unsafe {
            debug!("ProtectBlock.handle: setting up handler block");
            let p : *RustClosure = ::cast::transmute(&h);
            let prev = task::local_data::local_data_get(self.cond.key);
            HandleBlock { pb: self,
                          handler: @Handler{handle: *p, prev: prev} }
        }
    }
}


impl Condition {

    fn protect(&self, inner: &self/fn()) -> ProtectBlock/&self {
        unsafe {
            // transmutation to avoid copying non-copyable, should
            // be fixable by tracking closure pointees in regionck.
            debug!("Condition.protect: setting up protected block");
            let p : *RustClosure = ::cast::transmute(&inner);
            ProtectBlock { cond: self,
                           inner: *p }
        }
    }

    fn raise<T, U: Copy>(t:&T) -> U {
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
fn happiness_key(_x: @Handler) { }

#[cfg(test)]
fn sadness_key(_x: @Handler) { }

#[cfg(test)]
fn trouble(i: int) {
    // Condition should work as a const, just limitations in consts.
    let sadness_condition : Condition = Condition { key: sadness_key };
    debug!("trouble: raising conition");
    let j = sadness_condition.raise(&i);
    debug!("trouble: handler recovered with %d", j);
}

#[test]
fn test1() {

    let sadness_condition : Condition = Condition { key: sadness_key };

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
    let sadness_condition : Condition = Condition { key: sadness_key };

    let mut inner_trapped = false;

    let b = do sadness_condition.protect {
        debug!("nested_test_inner: in protected block");
        trouble(1);
    };

    do b.handle |_j:&int| {
        debug!("nested_test_inner: in handler");
        inner_trapped = true;
        0
    };

    assert inner_trapped;
}

#[test]
fn nested_test_outer() {

    let sadness_condition : Condition = Condition { key: sadness_key };

    let mut outer_trapped = false;

    let b = do sadness_condition.protect {
        debug!("nested_test_outer: in protected block");
        nested_test_inner();
        trouble(1);
    };

    do b.handle |_j:&int| {
        debug!("nested_test_outer: in handler");
        outer_trapped = true;
        0
    };

    assert outer_trapped;
}
