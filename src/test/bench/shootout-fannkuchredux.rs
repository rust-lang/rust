// Based on Isaac Gouy's fannkuchredux.csharp
use std;
import int;
import vec;

fn fannkuch(n: int) -> int {
    fn perm1init(i: uint) -> int { ret i as int; }

    let perm = vec::init_elt_mut(0, n as uint);
    let perm1 = vec::init_fn_mut(perm1init, n as uint);
    let count = vec::init_elt_mut(0, n as uint);
    let f = 0;
    let i = 0;
    let k = 0;
    let r = 0;
    let flips = 0;
    let nperm = 0;
    let checksum = 0;
    r = n;
    while r > 0 {
        i = 0;
        while r != 1 { count[r - 1] = r; r -= 1; }
        while i < n { perm[i] = perm1[i]; i += 1; }
        // Count flips and update max and checksum

        f = 0;
        k = perm[0];
        while k != 0 {
            i = 0;
            while 2 * i < k {
                let t = perm[i];
                perm[i] = perm[k - i];
                perm[k - i] = t;
                i += 1;
            }
            k = perm[0];
            f += 1;
        }
        if f > flips { flips = f; }
        if nperm & 0x1 == 0 { checksum += f; } else { checksum -= f; }
        // Use incremental change to generate another permutation

        let go = true;
        while go {
            if r == n {
                std::io::println(#fmt("%d", checksum));
                ret flips;
            }
            let p0 = perm1[0];
            i = 0;
            while i < r { let j = i + 1; perm1[i] = perm1[j]; i = j; }
            perm1[r] = p0;
            count[r] -= 1;
            if count[r] > 0 { go = false; } else { r += 1; }
        }
        nperm += 1;
    }
    ret flips;
}

fn main(args: [str]) {
    let n = if vec::len(args) == 2u {
        int::from_str(args[1])
    } else {
        8
    };
    std::io::println(#fmt("Pfannkuchen(%d) = %d", n, fannkuch(n)));
}
