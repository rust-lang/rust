struct cat {
  name: ~str,
}

impl cat : Drop {
    #[cat_dropper]
    /**
       Actually, cats don't always land on their feet when you drop them.
    */
    fn finalize(&self) {
        error!("%s landed on hir feet", self.name);
    }
}

#[cat_maker]
/**
Maybe it should technically be a kitten_maker.
*/
fn cat(name: ~str) -> cat {
    cat {
        name: name
    }
}

fn main() {
  let _kitty = cat(~"Spotty");
}
