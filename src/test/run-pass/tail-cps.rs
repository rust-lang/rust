


// -*- rust -*-
fn checktrue(bool rs) -> bool { assert (rs); ret true; }

fn main() { auto k = checktrue; evenk(42, k); oddk(45, k); }

fn evenk(int n, fn(bool) -> bool  k) -> bool {
    log "evenk";
    log n;
    if (n == 0) { be k(true); } else { be oddk(n - 1, k); }
}

fn oddk(int n, fn(bool) -> bool  k) -> bool {
    log "oddk";
    log n;
    if (n == 0) { be k(false); } else { be evenk(n - 1, k); }
}