struct X(usize);

impl X {
    fn zap(&self) {
        //~^ HELP
        //~| SUGGESTION &mut self
        self.0 = 32;
        //~^ ERROR
    }
}

fn main() {
    let ref foo = 16;
    //~^ HELP
    //~| SUGGESTION ref mut foo
    *foo = 32;
    //~^ ERROR
    if let Some(ref bar) = Some(16) {
        //~^ HELP
        //~| SUGGESTION ref mut bar
        *bar = 32;
        //~^ ERROR
    }
    match 16 {
        ref quo => { *quo = 32; },
        //~^ ERROR
        //~| HELP
        //~| SUGGESTION ref mut quo
    }
}
