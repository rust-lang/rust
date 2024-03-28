macro_rules! dollar_dollar {
    () => {
        macro_rules! bar {
            ( $$( $$any:tt )* ) => { $$( $$any )* };
            //~^ ERROR meta-variable expressions are unstable
            //~| ERROR meta-variable expressions are unstable
            //~| ERROR meta-variable expressions are unstable
            //~| ERROR meta-variable expressions are unstable
        }
    };
}

fn main() {
}
