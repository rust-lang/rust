mod cat {
    struct Cat {
        priv meows: uint;
    }

    fn new_cat() -> Cat {
        Cat { meows: 52 }
    }
}

fn main() {
    let nyan = cat::new_cat();
    assert nyan.meows == 52;    //~ ERROR field `meows` is private
}
