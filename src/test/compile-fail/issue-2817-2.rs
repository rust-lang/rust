fn not_bool(f: fn(int) -> ~str) {}

fn main() {
    for uint::range(0, 100000) |_i| { //~ ERROR A for-loop body must return (), but
        false
    };
    for not_bool |_i| { //~ ERROR a `loop` function's last argument should return `bool`
        //~^ ERROR A for-loop body must return (), but
        ~"hi"
    };
    for uint::range(0, 100000) |_i| { //~ ERROR A for-loop body must return (), but
        ~"hi"
    };
    for not_bool() |_i| { //~ ERROR a `loop` function's last argument
    };
}