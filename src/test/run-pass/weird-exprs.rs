// Just a grab bag of stuff that you wouldn't want to actually write.

fn strange() -> bool { let _x: bool = ret true; }

fn funny() {
    fn f(_x: ()) { }
    f(ret);
}

fn what() {
    fn the(x: @mut bool) { ret while !*x { *x = true; }; }
    let i = @mut false;
    let dont = {||the(i)};
    dont();
    assert (*i);
}

fn zombiejesus() {
    loop {
        while (ret) {
            if (ret) {
                alt (ret) {
                    1 {
                        if (ret) {
                            ret
                        } else {
                            ret
                        }
                    }
                    _ { ret }
                };
            } else if (ret) {
                ret;
            }
        }
        if (ret) { break; }
    }
}

fn notsure() {
    let mut _x;
    let mut _y = (_x = 0) == (_x = 0);
    let mut _z = (_x <- 0) < (_x = 0);
    let _a = (_x += 0) == (_x = 0);
    let _b = (_y <-> _z) == (_y <-> _z);
}

fn hammertime() -> int {
    let _x = log(debug, true == (ret 0));
}

fn canttouchthis() -> uint {
    pure fn p() -> bool { true }
    let _a = (assert (true)) == (assert (p()));
    let _c = (assert (p())) == ();
    let _b: bool = (log(debug, 0) == (ret 0u));
}

fn angrydome() {
    loop { if break { } }
    let mut i = 0;
    loop { i += 1; if i == 1 { alt check again { 1 { } } } break; }
}

fn evil_lincoln() { let evil <- #debug("lincoln"); }

fn main() {
    strange();
    funny();
    what();
    zombiejesus();
    notsure();
    hammertime();
    canttouchthis();
    angrydome();
    evil_lincoln();
}
