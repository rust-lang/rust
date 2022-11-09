// error-pattern: not found in this scope

fn main() {
    x = x = x;
    x = y = y = y;
    x = y = y;
    x = x = y;
    x = x; // will suggest add `let`
    x = y // will suggest add `let`
}
