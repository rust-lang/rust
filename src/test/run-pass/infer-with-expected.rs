// Tests the passing down of expected types through boxing and
// wrapping in a record or tuple. (The a.x would complain about 'this
// type must be known in this context' if the passing down doesn't
// happen.)

fn eat_tup(_r: ~@(int, fn@({x: int, y: int}) -> int)) {}
fn eat_rec(_r: @~{a: int, b: fn@({x: int, y: int}) -> int}) {}

fn main() {
    eat_tup(~@(10, {|a| a.x}));
    eat_rec(@~{a: 10, b: {|a| a.x}});
}
