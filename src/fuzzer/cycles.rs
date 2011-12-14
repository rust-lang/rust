use std;
import vec;
import std::rand;
import option;

// random uint less than n
fn under(r : rand::rng, n : uint) -> uint { assert n != 0u; r.next() as uint % n }

// random choice from a vec
fn choice<T>(r : rand::rng, v : [T]) -> T { assert vec::len(v) != 0u; v[under(r, vec::len(v))] }

// 1 in n chance of being true
fn unlikely(r : rand::rng, n : uint) -> bool { under(r, n) == 0u }

tag maybe_pointy {
  no_pointy;
  yes_pointy(@pointy);
}

type pointy = {
  mutable x : maybe_pointy,
  mutable y : maybe_pointy,
  mutable z : fn()->()
};

fn allunder(n: uint, it: block(uint)) {
    let i: uint = 0u;
    while i < n { it(i); i += 1u; }
}

fn nopT(_x : @pointy) { }
fn nop() { }

fn test_cycles(r : rand::rng)
{
    const max : uint = 10u;

    let v : [mutable @pointy] = [mutable];
    allunder(max) {|i|
        v += [mutable @{ mutable x : no_pointy, mutable y : no_pointy, mutable z: nop }];
    }

    allunder(max) {|i|
        v[i].x = yes_pointy(v[under(r, max)]);
        v[i].y = yes_pointy(v[under(r, max)]);
        v[i].z = bind nopT(v[under(r, max)]);
    }

    // Drop refs one at a time
    allunder(max) {|i|
        v[i] = @{ mutable x : no_pointy, mutable y : no_pointy, mutable z: nop };
    }
}

fn main()
{
    let r = rand::mk_rng();
    test_cycles(r);
}
