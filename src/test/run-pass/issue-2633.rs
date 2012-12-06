struct cat {
    mut meow: fn@(),
}

fn cat() -> cat {
    cat {
        meow: fn@() { error!("meow"); }
    }
}

type kitty_info = {kitty: cat};

// Code compiles and runs successfully if we add a + before the first arg
fn nyan(kitty: cat, _kitty_info: kitty_info) {
    (kitty.meow)();
}

fn main() {
    let mut kitty = cat();
    nyan(copy kitty, {kitty: copy kitty});
}
