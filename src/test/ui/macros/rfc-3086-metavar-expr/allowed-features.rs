// check-pass

macro_rules! dollar_dollar {
    () => {
        macro_rules! bar {
            ( $$( $$any:tt )* ) => { $$( $$any )* };
        }
    };
}

fn main() {
}
