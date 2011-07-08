fn to_str(u64 n, uint radix) -> str {
    assert(0u < radix && radix <= 16u);

    auto r64 = radix as u64;
    
    fn digit(u64 n) -> str {
        ret alt (n) {
            case (0u64) { "0" }
            case (1u64) { "1" }
            case (2u64) { "2" }
            case (3u64) { "3" }
            case (4u64) { "4" }
            case (5u64) { "5" }
            case (6u64) { "6" }
            case (7u64) { "7" }
            case (8u64) { "8" }
            case (9u64) { "9" }
            case (10u64) { "a" }
            case (11u64) { "b" }
            case (12u64) { "c" }
            case (13u64) { "d" }
            case (14u64) { "e" }
            case (15u64) { "f" }
            case (_) { fail }
        };
    }

    if n == 0u64 { ret "0"; }

    auto s = "";

    while(n > 0u64) {
        s = digit(n % r64) + s;
        n /= r64;
    }
    ret s;
}

fn str(u64 n) -> str { ret to_str(n, 10u); }
