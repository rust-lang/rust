// Issue #570

const lsl : int = 1 << 2;
const add : int = 1 + 2;
const addf : float = 1.0f + 2.0f;
const not : int = !0;
const notb : bool = !true;
const neg : int = -(1);

fn main() {
    assert(lsl == 4);
    assert(add == 3);
    assert(addf == 3.0f);
    assert(not == -1);
    assert(notb == false);
    assert(neg == -1);
}
