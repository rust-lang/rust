// Just a grab bag of stuff that you wouldn't want to actually write.

fn strange() -> bool { let _x = ret true; }

fn funny() {
    fn f(_x: ()) { }
    f(ret);
}

fn what() {
    fn the(x: @mutable bool) { ret while !*x { *x = true; }; }
    let i = @mutable false;
    let dont = bind the(i);
    dont();
    assert (*i);
}

fn zombiejesus() {
    do  {
        while (ret) {
            if (ret) {
                alt (ret) {
                    _ {
                        if (ret) {
                            ret
                        } else {
                            ret
                        }
                    }
                };
            } else if (ret) {
                ret;
            }
        }
    } while ret
}

fn notsure() {
    let _x;
    let _y = (_x = 0) == (_x = 0);
    let _z = (_x <- 0) < (_x = 0);
    let _a = (_x += 0) == (_x = 0);
    let _b = (_y <-> _z) == (_y <-> _z);
}

fn hammertime() -> int {
    let _x = log(debug, true == (ret 0));
}

fn canttouchthis() -> uint {
    pure fn p() -> bool { true }
    let _a = (assert (true)) == (check (p()));
    let _c = (check (p())) == ();
    let _b = (log(debug, 0) == (ret 0u));
}

fn angrydome() {
    while true { if break { } }
    let i = 0;
    do  { i += 1; if i == 1 { alt cont { _ { } } } } while false
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
