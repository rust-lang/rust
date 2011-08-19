// xfail-pretty
// Just a grab bug of stuff that you wouldn't want to actualy write

fn strange() -> bool {
    let _x = ret true;
}

fn funny() {
    fn f(_x: ()) {}
    f(ret);
}

fn odd() {
    // FIXME: This doesn't compile
    // log ret;
}

fn what() {
    fn the(x: @mutable bool){
        ret while !*x { *x = true };
    }
    let i = @mutable false;
    let dont = bind the(i);
    dont();
    assert *i;
}

fn zombiejesus() {
    do { while (ret) { if (ret) {
        alt (ret) { _ {
          ret ? ret : ret
        }}
    }}} while ret;
}

fn notsure() {
    let _x;
    let _y = (_x = 0) == (_x = 0);
    let _z = (_x <- 0) < (_x = 0);
    let _a = (_x += 0) == (_x = 0);
    let _b = (_y <-> _z) == (_y <-> _z);
}

fn hammertime() -> int {
    // FIXME: Doesn't compile
    //let _x = log true == (ret 0);
    ret 0;
}

fn canttouchthis() -> uint {
    pred p() -> bool { true }
    let _a = (assert true) == (check p());
    let _c = (check p()) == ();
    let _b = (log 0) == (ret 0u);
}

fn angrydome() {
    while true {
        if (break) { }
    }
    let i = 0;
    do {
        i += 1;
        if i == 1 {
            alt cont { _ { } }
        }
    } while false;
}

fn evil_lincoln() {
    let evil <- log "lincoln";
}

fn main() {
    strange();
    funny();
    odd();
    what();
    zombiejesus();
    notsure();
    hammertime();
    canttouchthis();
    angrydome();
    evil_lincoln();
}
