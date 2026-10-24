struct Guard<'a> {
    f: Box<dyn Fn() + Send + 'a>,
}

fn scoped<'a, F: Fn() + Send + 'a>(f: F) -> Guard<'a> {
    Guard { f: Box::new(f) }
}

impl<'a> Guard<'a> {
    fn join(self) {}
}

fn main() {
    let bad = {
        let x = 1;
        let y = &x;
        //~^ ERROR `x` does not live long enough

        scoped(|| {
            let _z = y;
            //~^ ERROR `y` does not live long enough
        })
    };

    bad.join();
}
