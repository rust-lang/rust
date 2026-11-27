use std::clone::Share;
//~^ ERROR use of unstable library feature `share_trait`

#[derive(Clone)]
struct Alias;

impl Share for Alias {}
//~^ ERROR use of unstable library feature `share_trait`

fn share_generic<T: Share>(value: &T) -> T {
    //~^ ERROR use of unstable library feature `share_trait`
    value.share()
    //~^ ERROR use of unstable library feature `share_trait`
}

fn main() {
    let value = Alias;
    let _ = Share::share(&value);
    //~^ ERROR use of unstable library feature `share_trait`
}
