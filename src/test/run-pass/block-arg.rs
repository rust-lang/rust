// Check usage and precedence of block arguments in expressions:
fn main() {
    let v = ~[-1f, 0f, 1f, 2f, 3f];

    // Statement form does not require parentheses:
    do vec::iter(v) |i| {
        log(info, i);
    }

    // Usable at all:
    let mut any_negative = do vec::any(v) |e| { float::is_negative(e) };
    assert any_negative;

    // Higher precedence than assignments:
    any_negative = do vec::any(v) |e| { float::is_negative(e) };
    assert any_negative;

    // Higher precedence than unary operations:
    let abs_v = do vec::map(v) |e| { float::abs(e) };
    assert do vec::all(abs_v) |e| { float::is_nonnegative(e) };
    assert !do vec::any(abs_v) |e| { float::is_negative(e) };

    // Usable in funny statement-like forms:
    if !do vec::any(v) |e| { float::is_positive(e) } {
        assert false;
    }
    alt do vec::all(v) |e| { float::is_negative(e) } {
        true => { fail ~"incorrect answer."; }
        false => { }
    }
    alt 3 {
      _ if do vec::any(v) |e| { float::is_negative(e) } => {
      }
      _ => {
        fail ~"wrong answer.";
      }
    }


    // Lower precedence than binary operations:
    let w = do vec::foldl(0f, v) |x, y| { x + y } + 10f;
    let y = do vec::foldl(0f, v) |x, y| { x + y } + 10f;
    let z = 10f + do vec::foldl(0f, v) |x, y| { x + y };
    assert w == y;
    assert y == z;

    // In the tail of a block
    let w =
        if true { do vec::any(abs_v) |e| { float::is_nonnegative(e) } }
      else { false };
    assert w;
}
