tag mytype = rec(fn (&mytype i) -> int compute, int val);

fn compute(&mytype i) -> int {
    ret i.val + 20;
}

fn main() {
    auto myval = mytype(rec(compute=compute, val=30));
    assert(myval.compute(myval) == 50);
}
