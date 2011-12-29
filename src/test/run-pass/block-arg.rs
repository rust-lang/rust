// Check usage and precedence of block arguments in expressions:
fn main() {
    let v = [-1f, 0f, 1f, 2f, 3f];

    // Statement form does not require parentheses:
    vec::iter(v) { |i|
        log(info, i);
    }

    // Usable at all:
    let any_negative = vec::any(v) { |e| float::negative(e) };
    assert any_negative;

    // Higher precedence than assignments:
    any_negative = vec::any(v) { |e| float::negative(e) };
    assert any_negative;

    // Higher precedence than unary operations:
    let abs_v = vec::map(v) { |e| float::abs(e) };
    assert vec::all(abs_v) { |e| float::nonnegative(e) };
    assert !vec::any(abs_v) { |e| float::negative(e) };

    // Usable in funny statement-like forms:
    if !vec::any(v) { |e| float::positive(e) } {
        assert false;
    }
    alt vec::all(v) { |e| float::negative(e) } {
        true { fail "incorrect answer."; }
        false { }
    }
    alt 3 {
      _ when vec::any(v) { |e| float::negative(e) } {
      }
      _ {
        fail "wrong answer.";
      }
    }


    // Lower precedence than binary operations:
    let w = vec::foldl(0f, v, { |x, y| x + y }) + 10f;
    let y = vec::foldl(0f, v) { |x, y| x + y } + 10f;
    let z = 10f + vec::foldl(0f, v) { |x, y| x + y };
    assert w == y;
    assert y == z;

    // They are not allowed as the tail of a block without parentheses:
    let w =
      if true { vec::any(abs_v, { |e| float::nonnegative(e) }) }
      else { false };
    assert w;
}
