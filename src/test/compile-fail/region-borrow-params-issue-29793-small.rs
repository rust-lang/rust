// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #29793, small regression tests: do not let borrows of
// parameters to ever be returned (expanded with exploration of
// variations).

// CLOSURES

fn escaping_borrow_of_closure_params_1() {
    let g = |x: usize, y:usize| {
        let f = |t: bool| if t { x } else { y }; // (separate errors for `x` vs `y`)
        //~^ ERROR `x` does not live long enough
        //~| ERROR `y` does not live long enough
        //~| NOTE capture occurs here
        //~| NOTE capture occurs here
        //~| NOTE does not live long enough
        //~| NOTE does not live long enough
        //~| NOTE values in a scope are dropped in the opposite order they are created
        //~| NOTE values in a scope are dropped in the opposite order they are created
        return f;
    };
    //~^ NOTE borrowed value dropped before borrower
    //~| NOTE borrowed value dropped before borrower

    // We delberately do not call `g`; this small version of the test,
    // after adding such a call, was (properly) rejected even when the
    // system still suffered from issue #29793.

    // g(10, 20)(true);
}

fn escaping_borrow_of_closure_params_2() {
    let g = |x: usize, y:usize| {
        let f = |t: bool| if t { x } else { y }; // (separate errors for `x` vs `y`)
        //~^ ERROR `x` does not live long enough
        //~| ERROR `y` does not live long enough
        //~| NOTE capture occurs here
        //~| NOTE capture occurs here
        //~| NOTE does not live long enough
        //~| NOTE does not live long enough
        //~| NOTE values in a scope are dropped in the opposite order they are created
        //~| NOTE values in a scope are dropped in the opposite order they are created
        f
    };
    //~^ NOTE borrowed value dropped before borrower
    //~| NOTE borrowed value dropped before borrower

    // (we don't call `g`; see above)
}

fn move_of_closure_params() {
    let g = |x: usize, y:usize| {
        let f = move |t: bool| if t { x } else { y };
        f;
    };
    // (this code is fine, so lets go ahead and ensure rustc accepts call of `g`)
    (g(1,2));
}

fn ok_borrow_of_fn_params(a: usize, b:usize) {
    let g = |x: usize, y:usize| {
        let f = |t: bool| if t { a } else { b };
        return f;
    };
    // (this code is fine, so lets go ahead and ensure rustc accepts call of `g`)
    (g(1,2))(true);
}

// TOP-LEVEL FN'S

fn escaping_borrow_of_fn_params_1() {
    fn g<'a>(x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a> {
        let f = |t: bool| if t { x } else { y }; // (separate errors for `x` vs `y`)
        //~^ ERROR E0373
        //~| NOTE `x` is borrowed here
        //~| NOTE may outlive borrowed value `x`
        //~| ERROR E0373
        //~| NOTE `y` is borrowed here
        //~| NOTE may outlive borrowed value `y`
        return Box::new(f);
    };

    // (we don't call `g`; see above)
}

fn escaping_borrow_of_fn_params_2() {
    fn g<'a>(x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a> {
        let f = |t: bool| if t { x } else { y }; // (separate errors for `x` vs `y`)
        //~^ ERROR E0373
        //~| NOTE `x` is borrowed here
        //~| NOTE may outlive borrowed value `x`
        //~| ERROR E0373
        //~| NOTE `y` is borrowed here
        //~| NOTE may outlive borrowed value `y`
        Box::new(f)
    };

    // (we don't call `g`; see above)
}

fn move_of_fn_params() {
    fn g<'a>(x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a> {
        let f = move |t: bool| if t { x } else { y };
        return Box::new(f);
    };
    // (this code is fine, so lets go ahead and ensure rustc accepts call of `g`)
    (g(1,2))(true);
}

// INHERENT METHODS

fn escaping_borrow_of_method_params_1() {
    struct S;
    impl S {
        fn g<'a>(&self, x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a> {
            let f = |t: bool| if t { x } else { y }; // (separate errors for `x` vs `y`)
            //~^ ERROR E0373
            //~| NOTE `x` is borrowed here
            //~| NOTE may outlive borrowed value `x`
            //~| ERROR E0373
            //~| NOTE `y` is borrowed here
            //~| NOTE may outlive borrowed value `y`
            return Box::new(f);
        }
    }

    // (we don't call `g`; see above)
}

fn escaping_borrow_of_method_params_2() {
    struct S;
    impl S {
        fn g<'a>(&self, x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a> {
            let f = |t: bool| if t { x } else { y }; // (separate errors for `x` vs `y`)
            //~^ ERROR E0373
            //~| NOTE `x` is borrowed here
            //~| NOTE may outlive borrowed value `x`
            //~| ERROR E0373
            //~| NOTE `y` is borrowed here
            //~| NOTE may outlive borrowed value `y`
            Box::new(f)
        }
    }
    // (we don't call `g`; see above)
}

fn move_of_method_params() {
    struct S;
    impl S {
        fn g<'a>(&self, x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a> {
            let f = move |t: bool| if t { x } else { y };
            return Box::new(f);
        }
    }
    // (this code is fine, so lets go ahead and ensure rustc accepts call of `g`)
    (S.g(1,2))(true);
}

// TRAIT IMPL METHODS

fn escaping_borrow_of_trait_impl_params_1() {
    trait T { fn g<'a>(&self, x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a>; }
    struct S;
    impl T for S {
        fn g<'a>(&self, x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a> {
            let f = |t: bool| if t { x } else { y }; // (separate errors for `x` vs `y`)
            //~^ ERROR E0373
            //~| NOTE `x` is borrowed here
            //~| NOTE may outlive borrowed value `x`
            //~| ERROR E0373
            //~| NOTE `y` is borrowed here
            //~| NOTE may outlive borrowed value `y`
            return Box::new(f);
        }
    }

    // (we don't call `g`; see above)
}

fn escaping_borrow_of_trait_impl_params_2() {
    trait T { fn g<'a>(&self, x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a>; }
    struct S;
    impl T for S {
        fn g<'a>(&self, x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a> {
            let f = |t: bool| if t { x } else { y }; // (separate errors for `x` vs `y`)
            //~^ ERROR E0373
            //~| NOTE `x` is borrowed here
            //~| NOTE may outlive borrowed value `x`
            //~| ERROR E0373
            //~| NOTE `y` is borrowed here
            //~| NOTE may outlive borrowed value `y`
            Box::new(f)
        }
    }
    // (we don't call `g`; see above)
}

fn move_of_trait_impl_params() {
    trait T { fn g<'a>(&self, x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a>; }
    struct S;
    impl T for S {
        fn g<'a>(&self, x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a> {
            let f = move |t: bool| if t { x } else { y };
            return Box::new(f);
        }
    }
    // (this code is fine, so lets go ahead and ensure rustc accepts call of `g`)
    (S.g(1,2))(true);
}

// TRAIT DEFAULT METHODS

fn escaping_borrow_of_trait_default_params_1() {
    struct S;
    trait T {
        fn g<'a>(&self, x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a> {
            let f = |t: bool| if t { x } else { y }; // (separate errors for `x` vs `y`)
            //~^ ERROR E0373
            //~| NOTE `x` is borrowed here
            //~| NOTE may outlive borrowed value `x`
            //~| ERROR E0373
            //~| NOTE `y` is borrowed here
            //~| NOTE may outlive borrowed value `y`
            return Box::new(f);
        }
    }
    impl T for S {}
    // (we don't call `g`; see above)
}

fn escaping_borrow_of_trait_default_params_2() {
    struct S;
    trait T {
        fn g<'a>(&self, x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a> {
            let f = |t: bool| if t { x } else { y }; // (separate errors for `x` vs `y`)
            //~^ ERROR E0373
            //~| NOTE `x` is borrowed here
            //~| NOTE may outlive borrowed value `x`
            //~| ERROR E0373
            //~| NOTE `y` is borrowed here
            //~| NOTE may outlive borrowed value `y`
            Box::new(f)
        }
    }
    impl T for S {}
    // (we don't call `g`; see above)
}

fn move_of_trait_default_params() {
    struct S;
    trait T {
        fn g<'a>(&self, x: usize, y:usize) -> Box<Fn(bool) -> usize + 'a> {
            let f = move |t: bool| if t { x } else { y };
            return Box::new(f);
        }
    }
    impl T for S {}
    // (this code is fine, so lets go ahead and ensure rustc accepts call of `g`)
    (S.g(1,2))(true);
}

fn main() { }

