fn main() {
    let mut x = 0;
    
    loop foo: {
        loop bar: {
            loop quux: {
                if 1 == 2 {
                    break foo;
                }
                else {
                    break bar;
                }
            }
            loop foo;
        }
        x = 42;
        break;
    }

    error!("%?", x);
    assert(x == 42);
}