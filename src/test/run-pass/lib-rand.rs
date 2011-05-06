// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

use std;
import std.Rand;

fn main() {
  let Rand.rng r1 = Rand.mk_rng();
  log r1.next();
  log r1.next();
  {
    auto r2 = Rand.mk_rng();
    log r1.next();
    log r2.next();
    log r1.next();
    log r1.next();
    log r2.next();
    log r2.next();
    log r1.next();
    log r1.next();
    log r1.next();
    log r2.next();
    log r2.next();
    log r2.next();
  }
  log r1.next();
  log r1.next();
}
