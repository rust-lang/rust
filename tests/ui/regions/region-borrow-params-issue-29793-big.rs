// Issue #29793, big regression test: do not let borrows of
// parameters to ever be returned (expanded with exploration of
// variations).
//
// This is the version of the test that actually exposed unsound
// behavior (because the improperly accepted closure was actually
// able to be invoked).

struct WrapA<F>(Option<F>);

impl<F> WrapA<F> {
    fn new() -> WrapA<F> {
        WrapA(None)
    }
    fn set(mut self, f: F) -> Self {
        self.0 = Some(f);
        self
    }
}

struct WrapB<F>(Option<F>);

impl<F> WrapB<F> {
    fn new() -> WrapB<F> {
        WrapB(None)
    }
    fn set(mut self, f: F) -> Self {
        self.0 = Some(f);
        self
    }
}

trait DoStuff : Sized {
    fn handle(self);
}

impl<F, T> DoStuff for WrapA<F>
    where F: FnMut(usize, usize) -> T, T: DoStuff {
        fn handle(mut self) {
            if let Some(ref mut f) = self.0 {
                let x = f(1, 2);
                let _foo = [0usize; 16];
                x.handle();
            }
        }
    }

impl<F> DoStuff for WrapB<F> where F: FnMut(bool) -> usize {
    fn handle(mut self) {
        if let Some(ref mut f) = self.0 {
            println!("{}", f(true));
        }
    }
}

impl<F, T> WrapA<F>
    where F: FnMut(usize, usize) -> T, T: DoStuff {
        fn handle_ref(&mut self) {
            if let Some(ref mut f) = self.0 {
                let x = f(1, 2);
            }
        }
    }

fn main() {
    let mut w = WrapA::new().set(|x: usize, y: usize| {
        WrapB::new().set(|t: bool| if t { x } else { y }) // (separate errors for `x` vs `y`)
            //~^ ERROR closure may outlive the current function
            //~| ERROR closure may outlive the current function
    });

    w.handle(); // This works
    // w.handle_ref(); // This doesn't
}
