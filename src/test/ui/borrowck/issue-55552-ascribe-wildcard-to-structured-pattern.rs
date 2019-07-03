// build-pass (FIXME(62277): could be check-pass?)

// rust-lang/rust#55552: The strategy pnkfelix landed in PR #55274
// (for ensuring that NLL respects user-provided lifetime annotations)
// did not handle the case where the ascribed type has some expliit
// wildcards (`_`) mixed in, and it caused an internal compiler error
// (ICE).
//
// This test is just checking that we do not ICE when such things
// occur.

struct X;
struct Y;
struct Z;

struct Pair { x: X, y: Y }

pub fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
where A: FnOnce() -> RA + Send,
      B: FnOnce() -> RB + Send,
      RA: Send,
      RB: Send
{
    (oper_a(), oper_b())
}

fn main() {
    let ((_x, _y), _z): (_, Z) = join(|| (X, Y), || Z);

    let (Pair { x: _x, y: _y }, Z): (_, Z) = join(|| Pair { x: X, y: Y }, || Z);
}
