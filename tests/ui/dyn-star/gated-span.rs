macro_rules! t {
    ($t:ty) => {}
}

t!(dyn* Send);
//~^ ERROR `dyn*` trait objects are experimental

fn main() {}
