

// Based on Isaac Gouy's fannkuchredux.csharp
use std;
import std::int;
import std::vec;

fn fannkuch(int n) -> int {
    fn perm1init(uint i) -> int { ret i as int; }
    auto perm1init_ = perm1init; // Rustboot workaround

    auto perm = vec::init_elt_mut(0, n as uint);
    auto perm1 = vec::init_fn_mut(perm1init_, n as uint);
    auto count = vec::init_elt_mut(0, n as uint);
    auto f = 0;
    auto i = 0;
    auto k = 0;
    auto r = 0;
    auto flips = 0;
    auto nperm = 0;
    auto checksum = 0;
    r = n;
    while (r > 0) {
        i = 0;
        while (r != 1) { count.(r - 1) = r; r -= 1; }
        while (i < n) { perm.(i) = perm1.(i); i += 1; }
        // Count flips and update max and checksum

        f = 0;
        k = perm.(0);
        while (k != 0) {
            i = 0;
            while (2 * i < k) {
                auto t = perm.(i);
                perm.(i) = perm.(k - i);
                perm.(k - i) = t;
                i += 1;
            }
            k = perm.(0);
            f += 1;
        }
        if (f > flips) { flips = f; }
        if (nperm & 0x1 == 0) { checksum += f; } else { checksum -= f; }
        // Use incremental change to generate another permutation

        auto go = true;
        while (go) {
            if (r == n) { log checksum; ret flips; }
            auto p0 = perm1.(0);
            i = 0;
            while (i < r) { auto j = i + 1; perm1.(i) = perm1.(j); i = j; }
            perm1.(r) = p0;
            count.(r) -= 1;
            if (count.(r) > 0) { go = false; } else { r += 1; }
        }
        nperm += 1;
    }
    ret flips;
}

fn main(vec[str] args) {
    auto n = 7;
    log #fmt("Pfannkuchen(%d) = %d", n, fannkuch(n));
}