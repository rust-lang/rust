// pp-exact - Make sure we actually print the attributes

struct cat {
    #[cat_dropper]
    drop { error!("%s landed on hir feet",self.name); }
    name: ~str,
}

#[cat_maker]
fn cat(name: ~str) -> cat { cat{name: name,} }

fn main() { let _kitty = cat(~"Spotty"); }
