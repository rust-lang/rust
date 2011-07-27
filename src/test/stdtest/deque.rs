

// -*- rust -*-
use std;
import std::deque;

#[test]
fn test_simple() {
    let d: deque::t[int] = deque::create[int]();
    assert (d.size() == 0u);
    d.add_front(17);
    d.add_front(42);
    d.add_back(137);
    assert (d.size() == 3u);
    d.add_back(137);
    assert (d.size() == 4u);
    log d.peek_front();
    assert (d.peek_front() == 42);
    log d.peek_back();
    assert (d.peek_back() == 137);
    let i: int = d.pop_front();
    log i;
    assert (i == 42);
    i = d.pop_back();
    log i;
    assert (i == 137);
    i = d.pop_back();
    log i;
    assert (i == 137);
    i = d.pop_back();
    log i;
    assert (i == 17);
    assert (d.size() == 0u);
    d.add_back(3);
    assert (d.size() == 1u);
    d.add_front(2);
    assert (d.size() == 2u);
    d.add_back(4);
    assert (d.size() == 3u);
    d.add_front(1);
    assert (d.size() == 4u);
    log d.get(0);
    log d.get(1);
    log d.get(2);
    log d.get(3);
    assert (d.get(0) == 1);
    assert (d.get(1) == 2);
    assert (d.get(2) == 3);
    assert (d.get(3) == 4);
}

fn test_boxes(a: @int, b: @int, c: @int, d: @int) {
    let deq: deque::t[@int] = deque::create[@int]();
    assert (deq.size() == 0u);
    deq.add_front(a);
    deq.add_front(b);
    deq.add_back(c);
    assert (deq.size() == 3u);
    deq.add_back(d);
    assert (deq.size() == 4u);
    assert (deq.peek_front() == b);
    assert (deq.peek_back() == d);
    assert (deq.pop_front() == b);
    assert (deq.pop_back() == d);
    assert (deq.pop_back() == c);
    assert (deq.pop_back() == a);
    assert (deq.size() == 0u);
    deq.add_back(c);
    assert (deq.size() == 1u);
    deq.add_front(b);
    assert (deq.size() == 2u);
    deq.add_back(d);
    assert (deq.size() == 3u);
    deq.add_front(a);
    assert (deq.size() == 4u);
    assert (deq.get(0) == a);
    assert (deq.get(1) == b);
    assert (deq.get(2) == c);
    assert (deq.get(3) == d);
}

type eqfn[T] = fn(&T, &T) -> bool ;

fn test_parameterized[T](e: eqfn[T], a: &T, b: &T, c: &T, d: &T) {
    let deq: deque::t[T] = deque::create[T]();
    assert (deq.size() == 0u);
    deq.add_front(a);
    deq.add_front(b);
    deq.add_back(c);
    assert (deq.size() == 3u);
    deq.add_back(d);
    assert (deq.size() == 4u);
    assert (e(deq.peek_front(), b));
    assert (e(deq.peek_back(), d));
    assert (e(deq.pop_front(), b));
    assert (e(deq.pop_back(), d));
    assert (e(deq.pop_back(), c));
    assert (e(deq.pop_back(), a));
    assert (deq.size() == 0u);
    deq.add_back(c);
    assert (deq.size() == 1u);
    deq.add_front(b);
    assert (deq.size() == 2u);
    deq.add_back(d);
    assert (deq.size() == 3u);
    deq.add_front(a);
    assert (deq.size() == 4u);
    assert (e(deq.get(0), a));
    assert (e(deq.get(1), b));
    assert (e(deq.get(2), c));
    assert (e(deq.get(3), d));
}

tag taggy { one(int); two(int, int); three(int, int, int); }

tag taggypar[T] { onepar(int); twopar(int, int); threepar(int, int, int); }

type reccy = {x: int, y: int, t: taggy};

#[test]
fn test() {
    fn inteq(a: &int, b: &int) -> bool { ret a == b; }
    fn intboxeq(a: &@int, b: &@int) -> bool { ret a == b; }
    fn taggyeq(a: &taggy, b: &taggy) -> bool {
        alt a {
          one(a1) { alt b { one(b1) { ret a1 == b1; } _ { ret false; } } }
          two(a1, a2) {
            alt b {
              two(b1, b2) { ret a1 == b1 && a2 == b2; }
              _ { ret false; }
            }
          }
          three(a1, a2, a3) {
            alt b {
              three(b1, b2, b3) { ret a1 == b1 && a2 == b2 && a3 == b3; }
              _ { ret false; }
            }
          }
        }
    }
    fn taggypareq[T](a: &taggypar[T], b: &taggypar[T]) -> bool {
        alt a {
          onepar[T](a1) {
            alt b { onepar[T](b1) { ret a1 == b1; } _ { ret false; } }
          }
          twopar[T](a1, a2) {
            alt b {
              twopar[T](b1, b2) { ret a1 == b1 && a2 == b2; }
              _ { ret false; }
            }
          }
          threepar[T](a1, a2, a3) {
            alt b {
              threepar[T](b1, b2, b3) {
                ret a1 == b1 && a2 == b2 && a3 == b3;
              }
              _ { ret false; }
            }
          }
        }
    }
    fn reccyeq(a: &reccy, b: &reccy) -> bool {
        ret a.x == b.x && a.y == b.y && taggyeq(a.t, b.t);
    }
    log "*** test boxes";
    test_boxes(@5, @72, @64, @175);
    log "*** end test boxes";
    log "test parameterized: int";
    let eq1: eqfn[int] = inteq;
    test_parameterized[int](eq1, 5, 72, 64, 175);
    log "*** test parameterized: @int";
    let eq2: eqfn[@int] = intboxeq;
    test_parameterized[@int](eq2, @5, @72, @64, @175);
    log "*** end test parameterized @int";
    log "test parameterized: taggy";
    let eq3: eqfn[taggy] = taggyeq;
    test_parameterized[taggy](eq3, one(1), two(1, 2), three(1, 2, 3),
                              two(17, 42));
    /*
     * FIXME: Segfault.  Also appears to be caused only after upcall_grow_task

    log "*** test parameterized: taggypar[int]";
    let eqfn[taggypar[int]] eq4 = taggypareq[int];
    test_parameterized[taggypar[int]](eq4,
                                      onepar[int](1),
                                      twopar[int](1, 2),
                                      threepar[int](1, 2, 3),
                                      twopar[int](17, 42));
    log "*** end test parameterized: taggypar[int]";

     */

    log "*** test parameterized: reccy";
    let reccy1: reccy = {x: 1, y: 2, t: one(1)};
    let reccy2: reccy = {x: 345, y: 2, t: two(1, 2)};
    let reccy3: reccy = {x: 1, y: 777, t: three(1, 2, 3)};
    let reccy4: reccy = {x: 19, y: 252, t: two(17, 42)};
    let eq5: eqfn[reccy] = reccyeq;
    test_parameterized[reccy](eq5, reccy1, reccy2, reccy3, reccy4);
    log "*** end test parameterized: reccy";
    log "*** done";
}