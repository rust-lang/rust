use std;
import std::rand;
import vec;

// random uint less than n
fn under(r : rand::rng, n : uint) -> uint { assert n != 0u; r.next() as uint % n }

// random choice from a vec
fn choice<T>(r : rand::rng, v : [T]) -> T { assert vec::len(v) != 0u; v[under(r, vec::len(v))] }

// 1 in n chance of being true
fn unlikely(r : rand::rng, n : uint) -> bool { under(r, n) == 0u }

// shuffle a vec in place
fn shuffle<T>(r : rand::rng, &v : [mutable T]) {
    let i = vec::len(v);
    while i >= 2u {
        // Loop invariant: elements with index >= i have been locked in place.
        i -= 1u;
        vec::swap(v, i, under(r, i + 1u)); // Lock element i in place.
    }
}

// create a shuffled copy of a vec
fn shuffled<T>(r : rand::rng, v : [T]) -> [T] {
    let w = vec::to_mut(v);
    shuffle(r, w);
    vec::from_mut(w) // Shouldn't this happen automatically?
}

// sample from a population without replacement
//fn sample<T>(r : rand::rng, pop : [T], k : uint) -> [T] { fail }

// Two ways to make a weighted choice.
// * weighted_choice is O(number of choices) time
// * weighted_vec is O(total weight) space
type weighted<T> = { weight: uint, item: T };
fn weighted_choice<T>(r : rand::rng, v : [weighted<T>]) -> T {
    assert vec::len(v) != 0u;
    let total = 0u;
    for {weight: weight, item: _} in v {
        total += weight;
    }
    assert total >= 0u;
    let chosen = under(r, total);
    let so_far = 0u;
    for {weight: weight, item: item} in v {
        so_far += weight;
        if so_far > chosen {
            ret item;
        }
    }
    std::util::unreachable();
}

fn weighted_vec<T>(v : [weighted<T>]) -> [T] {
    let r = [];
    for {weight: weight, item: item} in v {
        let i = 0u;
        while i < weight {
            r += [item];
            i += 1u;
        }
    }
    r
}

fn main()
{
    let r = rand::mk_rng();

    log_full(core::error, under(r, 5u));
    log_full(core::error, choice(r, [10, 20, 30]));
    log_full(core::error, if unlikely(r, 5u) { "unlikely" } else { "likely" });

    let a = [mutable 1, 2, 3];
    shuffle(r, a);
    log_full(core::error, a);

    let i = 0u;
    let v = [
        {weight:1u, item:"low"},
        {weight:8u, item:"middle"},
        {weight:1u, item:"high"}
    ];
    let w = weighted_vec(v);

    while i < 1000u {
        log_full(core::error, "Immed: " + weighted_choice(r, v));
        log_full(core::error, "Fast: " + choice(r, w));
        i += 1u;
    }
}
