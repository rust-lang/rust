struct Session {
    opts: u8,
}

fn main() {
    let sess: &Session = &Session { opts: 0 };
    (sess as *const Session).opts; //~ ERROR no field `opts` on type `*const Session`
}
