// edition:2021

// regression test for #112056

struct Spooky<'b> {
    owned: Option<&'static u32>,
    borrowed: &'b &'static u32,
}

impl<'b> Spooky<'b> {
    fn create_self_reference<'a>(&'a mut self) {
        let mut closure = || {
            if let Some(owned) = &self.owned {
                let borrow: &'a &'static u32 = owned;
                self.borrowed = borrow;
                //~^ ERROR: lifetime may not live long enough
            }
        };
        closure();
    }
}

fn main() {
    let mut spooky: Spooky<'static> = Spooky {
        owned: Some(&1),
        borrowed: &&1,
    };
    spooky.create_self_reference();
    spooky.owned = None;
    println!("{}", **spooky.borrowed);
}
