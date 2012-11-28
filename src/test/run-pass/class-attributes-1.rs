// pp-exact - Make sure we actually print the attributes

struct cat {
    name: ~str,
}

impl cat: Drop {
    #[cat_dropper]
    fn finalize(&self) { error!("%s landed on hir feet",self.name); }
}


#[cat_maker]
fn cat(name: ~str) -> cat { cat{name: name,} }

fn main() { let _kitty = cat(~"Spotty"); }
