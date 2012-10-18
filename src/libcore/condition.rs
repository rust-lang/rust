// helper for transmutation, shown below.
type RustClosure = (int,int);

struct Condition {
    key: task::local_data::LocalDataKey<Handler>
}

struct Handler {
    // Handler should link to previous handler and
    // reinstall it when popped.
    handle: RustClosure
}


struct ProtectBlock {
    cond: &Condition,
    inner: RustClosure
}

struct PopHandler {
    cond: &Condition,
    drop {
        unsafe {
            task::local_data::local_data_pop(self.cond.key);
        }
    }
}

struct HandleBlock {
    pb: &ProtectBlock,
    handler: @Handler,
    drop {
        unsafe {
            task::local_data::local_data_set(self.pb.cond.key,
                                             self.handler);
            let _pop = PopHandler { cond: self.pb.cond };
            // transmutation to avoid copying non-copyable, should
            // be fixable by tracking closure pointees in regionck.
            let f : &fn() = ::cast::transmute(self.pb.inner);
            f();
        }
    }
}

impl ProtectBlock {
    fn handle<T, U: Copy>(&self, h: &self/fn(&T) ->U) -> HandleBlock/&self {
        unsafe {
            let p : *RustClosure = ::cast::transmute(&h);
            HandleBlock { pb: self,
                          handler: @Handler{handle: *p} }
        }
    }
}


impl Condition {

    fn protect(&self, inner: &self/fn()) -> ProtectBlock/&self {
        unsafe {
            // transmutation to avoid copying non-copyable, should
            // be fixable by tracking closure pointees in regionck.
            let p : *RustClosure = ::cast::transmute(&inner);
            ProtectBlock { cond: self,
                           inner: *p } }
    }

    fn raise<T, U: Copy>(t:&T) -> U {
        unsafe {
            match task::local_data::local_data_get(self.key) {
                None => fail,
                Some(handler) => {
                    io::println("got handler");
                    let f : &fn(&T) -> U = ::cast::transmute(handler.handle);
                    f(t)
                }
            }
        }
    }
}


#[test]
fn happiness_key(_x: @Handler) { }

#[test]
fn sadness_key(_x: @Handler) { }

#[test]
fn trouble(i: int) {
    // Condition should work as a const, just limitations in consts.
    let sadness_condition : Condition = Condition { key: sadness_key };
    io::println("raising");
    let j = sadness_condition.raise(&i);
    io::println(fmt!("handler recovered with %d", j));
}

#[test]
fn test() {

    let sadness_condition : Condition = Condition { key: sadness_key };

    let mut i = 10;

    let b = do sadness_condition.protect {
        io::println("in protected block");
        trouble(1);
        trouble(2);
        trouble(3);
    };

    do b.handle |j| {
        i += *j;
        i
    };

    assert i == 16;
}