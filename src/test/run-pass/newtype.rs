enum mytype = {compute: extern fn(mytype) -> int, val: int};

fn compute(i: mytype) -> int { return i.val + 20; }

fn main() {
    let myval = mytype({compute: compute, val: 30});
    assert (myval.compute(myval) == 50);
}
