// error-pattern:Tried to deinitialize a variable declared in a different
fn force(f: &block() -> int) -> int { ret f(); }
fn main() {
    let x = @{x:17, y:2};
    let y = @{x:5, y:5};

    let f =  {|&i| log_err i; x <- y; ret 7; };
    assert (f(5) == 7);
    log_err x;
    log_err y;
}
